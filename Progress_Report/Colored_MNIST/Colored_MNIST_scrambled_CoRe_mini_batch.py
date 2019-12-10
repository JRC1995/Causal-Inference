import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
import random

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=390)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.0011)
parser.add_argument('--lr', type=float, default=0.0004898536566546834)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=200)
parser.add_argument('--penalty_weight', type=float, default=1000)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--grayscale_model', action='store_true')
flags = parser.parse_args()

print('Flags:')
for k, v in sorted(vars(flags).items()):
    print("\t{}: {}".format(k, v))


def mini_batch(data_X, data_Y, batch_size=64):

    batches_X = []
    batches_Y = []
    data_size = len(data_X)
    i = 0
    while i < data_size:
        batch_X = []
        batch_Y = []
        if i+batch_size > data_size:
            batch_size = data_size-i

        batch_X = data_X[i:i+batch_size]
        batch_Y = data_Y[i:i+batch_size]

        batches_X.append(batch_X)
        batches_Y.append(batch_Y)

        i += batch_size

    return batches_X, batches_Y


np.random.seed(101)

# Load MNIST, make train/val splits, and shuffle train set examples

mnist = datasets.MNIST('datasets/mnist', train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])

rng_state = np.random.get_state()
np.random.shuffle(mnist_train[0].numpy())
np.random.set_state(rng_state)
np.random.shuffle(mnist_train[1].numpy())

# Build environments


def make_environment(images, labels, e):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a-b).abs()  # Assumes both inputs are either 0 or 1
    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = (labels < 5).float()
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
    return {
        'images': (images.float() / 255.).cuda(),
        'labels': labels[:, None].cuda()
    }


envs = [
    make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
    make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
    make_environment(mnist_val[0], mnist_val[1], 0.9)
]

final_train_accs = []
final_test_accs = []

for restart in range(flags.n_restarts):

    global_step = 0

    # Define and instantiate the model

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            if flags.grayscale_model:
                lin1 = nn.Linear(14 * 14, flags.hidden_dim)
            else:
                lin1 = nn.Linear(2 * 14 * 14, flags.hidden_dim)
            lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
            lin3 = nn.Linear(flags.hidden_dim, 1)
            for lin in [lin1, lin2, lin3]:
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
            self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

        def forward(self, input):
            if flags.grayscale_model:
                out = input.view(input.shape[0], 2, 14 * 14).sum(dim=1)
            else:
                out = input.view(input.shape[0], 2 * 14 * 14)
            out = self._main(out)
            return out

    mlp = MLP().cuda()
    # Define loss function helpers

    def nll_fn(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')

    def mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    # CoRe

    def penalty_fn(hidden, targets):
        np_targets = targets.cpu().numpy()
        classes = np.unique(np_targets)
        total_var = 0.0
        total_labels = 0
        for label in classes:
            idx = np.where(np_targets == label)[0]
            cc_hidden = hidden[idx]  # class conditioned logits
            cc_mean = cc_hidden.mean(dim=0, keepdim=True)
            total_var = total_var+(cc_hidden-cc_mean).pow(2).mean()
            total_labels += 1
        return total_var/total_labels
    # Train loop

    def pretty_print(*values):
        col_width = 13

        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)
        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))

    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    def run(batches_X, batches_Y, train=True):

        total_nll = 0.0
        total_penalty = 0.0
        total_acc = 0.0

        global global_step

        iter = 0

        for batch_X, batch_Y in zip(batches_X, batches_Y):
            indices = [i for i in range(len(batches_X))]
            chosen_idx = np.random.choice(indices)
            logits_1 = mlp(batch_X)
            logits_2 = mlp(batches_X[chosen_idx])
            labels_1 = batch_Y
            labels_2 = batches_Y[chosen_idx]

            penalty = penalty_fn(logits_1, labels_1)

            acc = mean_accuracy(logits_1, labels_1)

            nll = torch.mean(nll_fn(logits_1, labels_1))

            weight_norm = torch.tensor(0.).cuda()
            for w in mlp.parameters():
                weight_norm += w.norm().pow(2)

            loss = nll.clone()
            loss += flags.l2_regularizer_weight * weight_norm
            penalty_weight = (flags.penalty_weight
                              if global_step >= flags.penalty_anneal_iters else 1.0)

            loss += penalty_weight * penalty

            if penalty_weight > 1.0:
                # Rescale the entire loss to keep gradients in a reasonable range
                loss /= penalty_weight

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if iter % 100 == 0:
                # print(penalty_weight)
                pretty_print(
                    np.int32(step),
                    np.int32(iter),
                    nll.detach().cpu().numpy(),
                    acc.detach().cpu().numpy(),
                    penalty.detach().cpu().numpy(),
                )

            total_nll += nll.item()
            total_acc += acc.item()
            total_penalty += penalty.item()

            iter += 1
            global_step += 1

        total_nll /= iter
        total_acc /= iter
        total_penalty /= iter

        return total_nll, total_acc, total_penalty

    data_X1 = envs[0]['images']
    data_X2 = envs[1]['images']
    data_Y1 = envs[0]['labels']
    data_Y2 = envs[1]['labels']
    data_X3 = envs[2]['images']
    data_Y3 = envs[2]['labels']

    train_data_X_list = data_X1.cpu().tolist()+data_X2.cpu().tolist()
    train_data_Y_list = data_Y1.cpu().tolist()+data_Y2.cpu().tolist()

    indices = [i for i in range(len(train_data_X_list))]
    random.shuffle(indices)

    train_X = []
    train_Y = []

    for i in indices:
        train_X.append(train_data_X_list[i])
        train_Y.append(train_data_Y_list[i])

    train_X = torch.tensor(train_X).cuda()
    train_Y = torch.tensor(train_Y).cuda()

    for step in range(flags.steps):

        print("Training")
        pretty_print('epoch', 'iter', 'nll', 'acc', 'penalty')

        train_batches_X, train_batches_Y = mini_batch(train_X, train_Y)
        test_batches_X, test_batches_Y = mini_batch(data_X3, data_Y3)

        indices = [i for i in range(len(train_batches_X))]
        random.shuffle(indices)

        train_batches_X = [train_batches_X[i] for i in indices]
        train_batches_Y = [train_batches_Y[i] for i in indices]

        train_nll, train_acc, train_penalty = run(train_batches_X, train_batches_Y)

        print("Testing")
        pretty_print('epoch', 'iter', 'nll', 'acc', 'penalty')

        _, test_acc, _ = run(test_batches_X, test_batches_Y, train=False)

        print("\n\n")

        pretty_print('epoch', 'train nll', 'train acc', 'train penalty', 'test acc')

        pretty_print(
            np.int32(step),
            np.float32(train_nll),
            np.float32(train_acc),
            np.float32(train_penalty),
            np.float32(test_acc)
        )

        print("\n\n")

    final_train_accs.append(train_acc)
    final_test_accs.append(test_acc)
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    print("\n\n")
