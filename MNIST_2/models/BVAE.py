from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models.Encoder_Decoder import ConvEncoder, ConvDecoder
from torch import nn, optim, autograd
import numpy as np

# https://github.com/pytorch/examples/blob/master/vae/main.py


class BVAE(nn.Module):
    def __init__(self):
        super(BVAE, self).__init__()

        self.z_dim = 20
        self.hidden_dim = 20
        self.C = 3
        self.classes_num = 10
        #self.zeros = torch.tensor(0.0).cuda()

        self.Encoder = ConvEncoder(self.C, 2*self.z_dim)
        self.mu = nn.Linear(2*self.z_dim, self.z_dim)
        self.logvar = nn.Linear(2*self.z_dim, self.z_dim)
        self.Decoder = ConvDecoder(self.C, self.z_dim)
        self.classifier = nn.Sequential(nn.Linear(self.z_dim, self.hidden_dim),
                                        nn.ReLU(True),
                                        nn.Linear(self.hidden_dim, self.classes_num))  # 1 == class_num

    def encode(self, x):
        N, C, H, W = x.size()
        h0, h = self.Encoder(x)
        h = F.relu(h)
        h = h.view(N, 2*self.z_dim)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return h0, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        N, _ = z.size()
        z = z.view(N, self.z_dim, 1, 1)
        h = self.Decoder(z)
        return h

    def forward(self, x):
        h0, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return h0, self.classifier(z), self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def ELBO(recon_x, x, mu, logvar, beta=4, epoch=0, iter=0, model='vanilla', save_image_decision=True):

    if save_image_decision:

        save_image(recon_x,
                   'results/'+model+'_sample_' + str(epoch) + '_' + str(iter)+'.png')

    N, C, H, W = x.size()
    recon_x = recon_x.view(N, C*H*W)
    x = x.view(N, C*H*W)
    BCE = F.binary_cross_entropy(torch.sigmoid(recon_x), x, reduction='mean')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

    return BCE + beta*KLD


def nll_fn(logits, y):
    return F.cross_entropy(logits, y, reduction='mean')


def ER_penalty_fn(hidden, targets):
    np_targets = targets.cpu().numpy()
    classes = np.unique(np_targets)
    total_var = 0.0
    total_labels = 0
    for label in classes:
        idx = np.where(np_targets == label)[0]
        cc_hidden = hidden[idx]  # class conditioned logits
        cc_mean = cc_hidden.mean(dim=0, keepdim=True)
        total_var = total_var+(cc_hidden-cc_mean).pow(2).sum(-1).mean()
        total_labels += 1
    return total_var/total_labels


def CoRe_penalty_fn(hidden, targets):
    np_targets = targets.cpu().numpy()
    classes = np.unique(np_targets)
    total_var = 0.0
    total_labels = 0
    for label in classes:
        idx = np.where(np_targets == label)[0]
        cc_hidden = hidden[idx]  # class conditioned logits
        cc_mean = cc_hidden.mean(dim=0, keepdim=True)
        total_var = total_var+(cc_hidden-cc_mean).mean()
        total_labels += 1
    return total_var/total_labels


def penalty_fn(logits, y, train=True):

    if train:

        scale = torch.tensor(1.).cuda().requires_grad_()

        loss = nll_fn(logits * scale, y)

        loss = loss.view(-1, 1)

        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        biased_grad_norm = torch.sum(grad*grad, dim=-1).mean()

        return biased_grad_norm

    else:
        return torch.tensor(0.0).cuda()


def mean_accuracy(logits, y):
    preds = torch.argmax(logits, dim=-1)
    return ((preds - y).abs() < 1e-2).float().mean()
