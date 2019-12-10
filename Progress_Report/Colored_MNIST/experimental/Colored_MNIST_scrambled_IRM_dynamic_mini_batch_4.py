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
parser.add_argument('--penalty_weight', type=float, default=9000)
parser.add_argument('--steps', type=int, default=30)
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


def mini_batch_penalty_based(data_X, data_Y, batch_size=2048):

    batches_X = []
    batches_Y = []
    data_size = len(data_X)
    indices = [id for id in range(data_size)]
    group_size = data_size//3
    middle_indices = indices[group_size:2*group_size]
    early_indices = indices[0:group_size]
    end_indices = indices[2*group_size:]

    middle_indices_dict = {i: 1 for i in middle_indices}
    early_indices_dict = {i: 1 for i in early_indices}
    end_indices_dict = {i: 1 for i in end_indices}

    i = 0
    while i < data_size:
        batch_X = []
        batch_Y = []
        penalty_batches_X = []
        penalty_batches_Y = []

        if i+(batch_size) > data_size:
            batch_size = (data_size-i)

        batch_X = data_X[i:i+batch_size]
        batch_Y = data_Y[i:i+batch_size]

        if i in early_indices_dict:
            choose_from_indices = early_indices
        elif i in middle_indices_dict:
            x = random.choice([0, 1])
            if x == 0:
                choose_from_indices = end_indices
            else:
                choose_from_indices = early_indices
        else:
            choose_from_indices = end_indices

        chosen_indices = random.sample(choose_from_indices, k=batch_size)
        penalty_batch_X = [data_X[id] for id in chosen_indices]
        penalty_batch_Y = [data_Y[id] for id in chosen_indices]

        batch_X = torch.tensor(batch_X).cuda()
        batch_Y = torch.tensor(batch_Y).cuda()

        penalty_batch_X = torch.tensor(penalty_batch_X).cuda()
        penalty_batch_Y = torch.tensor(penalty_batch_Y).cuda()

        # print(batch_X.size())

        batches_X.append(batch_X)
        batches_Y.append(batch_Y)

        penalty_batches_X.append(penalty_batch_X)
        penalty_batches_Y.append(penalty_batch_Y)

        i += batch_size

    return batches_X, batches_Y, penalty_batches_X, penalty_batches_Y


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

    def penalty_fn(logits_1, y_1):

        scale = torch.tensor(1.).cuda().requires_grad_()

        loss_1 = nll_fn(logits_1 * scale, y_1)

        loss_1 = loss_1.view(-1, 1)

        grad_norm = []

        grad = autograd.grad(loss_1.mean(), [scale], create_graph=True)[0]
        mean_grad_norm = torch.sum(grad*grad)

        return mean_grad_norm

    def penalty_fn_ex(logits_1, y_1, logits_2, y_2):

        scale = torch.tensor(1.).cuda().requires_grad_()

        loss_1 = nll_fn(logits_1 * scale, y_1)

        loss_1 = loss_1.view(-1, 1)

        loss_2 = nll_fn(logits_2 * scale, y_2)

        loss_2 = loss_2.view(-1, 1)

        loss = (loss_1.mean()+loss_2.mean())/2

        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        mean_grad_norm = torch.sum(grad*grad)

        return mean_grad_norm

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

    def penalty_compute_run(Xs, Ys):

        penalty_list = []
        i = 0

        for X, Y in zip(Xs, Ys):
            X = torch.tensor([X]).cuda()
            Y = torch.tensor([Y]).cuda()
            logits = mlp(X)
            labels = Y

            penalty = penalty_fn(logits, labels)

            penalty_list.append(penalty.item())

            i += 1

            if i % 5000 == 0:
                print("Step: {}, Penalty: {}".format(i, penalty.item()))

        return penalty_list

    def run(batches_X, batches_Y, penalty_batches_X=None, penalty_batches_Y=None, train=True):

        total_nll = 0.0
        total_penalty = 0.0
        total_acc = 0.0

        global global_step

        iter = 0
        c = 0

        for batch_X, batch_Y in zip(batches_X, batches_Y):
            indices = [i for i in range(len(batches_X))]
            chosen_idx = np.random.choice(indices)
            logits = mlp(batch_X)
            labels = batch_Y

            if train:
                logits = mlp(penalty_batches_X[c])
                labels = penalty_batches_Y[c]
                penalty = penalty_fn_ex(logits, labels, penalty_logits, penalty_labels)
            else:
                penalty = penalty_fn(logits, labels)

            acc = mean_accuracy(logits, labels)

            nll = torch.mean(nll_fn(logits, labels))

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

    train_X = data_X1.cpu().tolist()+data_X2.cpu().tolist()
    train_Y = data_Y1.cpu().tolist()+data_Y2.cpu().tolist()

    train_X_tensor = torch.tensor(train_X).cuda()
    train_Y_tensor = torch.tensor(train_Y).cuda()

    for step in range(flags.steps):

        if step < 5:

            print("\n\nPreparing Penalty List\n")

            penalty_list = penalty_compute_run(train_X, train_Y)
            loss_list = np.asarray(penalty_list)
            loss_list = loss_list.reshape((-1))
            loss_list_indices = np.argsort(loss_list)
            loss_list_indices = loss_list_indices.tolist()
            train_X_ = []
            train_Y_ = []
            for id in loss_list_indices:
                train_X_.append(train_X[id])
                train_Y_.append(train_Y[id])
            train_X = train_X_
            train_Y = train_Y_

        print("Training")
        pretty_print('epoch', 'iter', 'nll', 'acc', 'penalty')

        if step < 5:
            train_batches_X, train_batches_Y, train_penalty_batches_X, train_penalty_batches_Y = mini_batch_penalty_based(
                train_X, train_Y)
        else:
            train_batches_X, train_batches_Y = mini_batch(train_X_tensor, train_Y_tensor)
        test_batches_X, test_batches_Y = mini_batch(data_X3, data_Y3)

        indices = [i for i in range(len(train_batches_X))]
        random.shuffle(indices)

        train_batches_X = [train_batches_X[i] for i in indices]
        train_batches_Y = [train_batches_Y[i] for i in indices]

        train_penalty_batches_X = [train_penalty_batches_X[i] for i in indices]
        train_penalty_batches_Y = [train_penalty_batches_Y[i] for i in indices]

        train_nll, train_acc, train_penalty = run(
            train_batches_X, train_batches_Y, train_penalty_batches_X, train_penalty_batches_Y)

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
