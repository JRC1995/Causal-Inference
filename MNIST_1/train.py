import numpy as np
import random
import torch as T
from torch import nn, optim, autograd
from common_functions import bucket_and_batch, pretty_print
from models.BVAE import BVAE, ELBO, nll_fn, penalty_fn, ER_penalty_fn, mean_accuracy
import argparse

random.seed(101)
val_size = 5000
batch_size = 125
display_step = 100

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--l2', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--penalty_anneal_iters', type=int, default=15)
parser.add_argument('--ELBO_weight', type=float, default=0)
parser.add_argument('--penalty_weight', type=float, default=0)
parser.add_argument('--ER_penalty_weight', type=float, default=0)
parser.add_argument('--epochs', type=int, default=10)
flags = parser.parse_args()


train_X = np.load('datasets/train_x.npy')
train_Y = np.load('datasets/train_y.npy')
test_X = np.load('datasets/test_x.npy')
test_Y = np.load('datasets/test_y.npy')

idx = [id for id in range(train_X.shape[0])]
random.shuffle(idx)

train_X = [train_X[id] for id in idx]
train_Y = [train_Y[id] for id in idx]

val_X = train_X[0:val_size]
val_Y = train_Y[0:val_size]
train_X = train_X[val_size:]
train_Y = train_Y[val_size:]

train_X = np.asarray(train_X, np.float32)
train_Y = np.asarray(train_Y, np.int)
val_X = np.asarray(val_X, np.float32)
val_Y = np.asarray(val_Y, np.int)

train_Y = np.reshape(train_Y, (-1))
test_Y = np.reshape(test_Y, (-1))
val_Y = np.reshape(val_Y, (-1))


global_step = 0

model = BVAE().cuda()
optimizer = optim.Adam(model.parameters(), lr=flags.lr)


def run(batches_X, batches_Y, batches_counter_X, batches_counter_Y, train=True, epoch=0):

    global model
    global display_step
    global global_step

    if train:
        model_ = model.train()
    else:
        model_ = model.eval()

    total_loss = 0.0
    total_penalty = 0.0
    total_acc = 0.0
    iter = 0

    for batch_X, batch_Y, \
            batch_counter_X, batch_counter_Y in zip(batches_X, batches_Y,
                                                    batches_counter_X, batches_counter_Y):

        z, logits, batch_recon_X, batch_mu, batch_logvar = model_(batch_X)
        labels = batch_Y

        penalty = penalty_fn(logits, labels, train=train)
        ER_penalty = ER_penalty_fn(z, labels)
        acc = mean_accuracy(logits, labels)
        nll = nll_fn(logits, labels)

        if flags.ELBO_weight > 0:
            save_image = True
        else:
            save_image = False
        elbo = ELBO(batch_recon_X, batch_X, batch_mu, batch_logvar,
                    epoch=epoch, iter=iter, save_image_decision=save_image, model='vanilla')

        weight_norm = T.tensor(0.).cuda()
        for w in model.parameters():
            weight_norm += w.norm().pow(2)

        # print(elbo.item())

        loss = nll*(1-flags.ELBO_weight) + elbo*flags.ELBO_weight
        loss = loss + flags.ER_penalty_weight*ER_penalty
        loss = loss + flags.l2 * weight_norm
        penalty_weight = (flags.penalty_weight
                          if global_step >= flags.penalty_anneal_iters else 1.0)
        loss = loss + penalty_weight * penalty

        if penalty_weight > 1.0:
            loss = loss/penalty_weight

        pure_loss = nll.item()*(1-flags.ELBO_weight) + elbo.item()*flags.ELBO_weight

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if iter % display_step == 0:
            pretty_print(
                np.int32(epoch),
                np.int32(iter),
                np.float32(pure_loss),
                acc.detach().cpu().numpy(),
                penalty.detach().cpu().numpy(),
            )

        total_loss += pure_loss
        total_acc += acc.item()
        total_penalty += penalty.item()

        if train:
            global_step += 1
        iter += 1

    total_loss /= iter
    total_acc /= iter
    total_penalty /= iter

    return total_loss, total_acc, total_penalty


train_batches_X, train_batches_Y, \
    train_batches_counter_X, train_batches_counter_Y = bucket_and_batch(
        train_X, train_Y, batch_size=batch_size)
val_batches_X, val_batches_Y, \
    val_batches_counter_X, val_batches_counter_Y = bucket_and_batch(
        val_X, val_Y, batch_size=batch_size)
test_batches_X, test_batches_Y,\
    test_batches_counter_X, test_batches_counter_Y = bucket_and_batch(
        test_X, test_Y, batch_size=batch_size)

print("hello")

test_accs = []
val_accs = []

for epoch in range(flags.epochs):
    print("Training")
    pretty_print('epoch', 'iter', 'loss', 'acc', 'penalty')
    train_loss, train_acc, train_penalty = \
        run(train_batches_X, train_batches_Y,
            train_batches_counter_X, train_batches_counter_Y, train=True, epoch=epoch)

    print("\n\n")

    print("Validating")
    pretty_print('epoch', 'iter', 'loss', 'acc', 'penalty')
    val_loss, val_acc, val_penalty = \
        run(val_batches_X, val_batches_Y,
            val_batches_counter_X, val_batches_counter_Y, train=False, epoch=epoch)

    print("\n\n")

    print("Testing")
    pretty_print('epoch', 'iter', 'loss', 'acc', 'penalty')
    with T.no_grad():
        test_loss, test_acc, test_penalty = \
            run(test_batches_X, test_batches_Y,
                test_batches_counter_X, test_batches_counter_Y, train=False, epoch=epoch)

    print("\n\nEPOCH SUMMARY\n\n")

    pretty_print('epoch', 'train loss', 'train penalty', 'train acc', 'val acc', 'test acc')

    pretty_print(
        np.int32(epoch),
        np.float32(train_loss),
        np.float32(train_penalty),
        np.float32(train_acc),
        np.float32(val_acc),
        np.float32(test_acc)
    )

    print("\n\n")

    test_accs.append(test_acc)
    val_accs.append(val_acc)

id = np.argmax(test_accs)

print("\n\n")
print("DEV ACCURACY: ", val_accs[id])
print("TEST ACCURACY: ", test_accs[id])
