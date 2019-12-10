import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd
import matplotlib.pyplot as plt
import random


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
    images = images.reshape((-1, 28, 28))
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

data_X1 = envs[0]['images'].cpu().numpy().tolist()
data_X2 = envs[1]['images'].cpu().numpy().tolist()
data_Y1 = envs[0]['labels'].cpu().numpy().tolist()
data_Y2 = envs[1]['labels'].cpu().numpy().tolist()
data_X3 = envs[2]['images'].cpu().numpy()
data_Y3 = envs[2]['labels'].cpu().numpy()

train_X = np.asarray(data_X1 + data_X2)
train_Y = np.asarray(data_Y1 + data_Y2)


np.save('datasets/train_x.npy', train_X)
np.save('datasets/train_y.npy', train_Y)
np.save('datasets/test_x.npy', data_X3)
np.save('datasets/test_y.npy', data_Y3)
