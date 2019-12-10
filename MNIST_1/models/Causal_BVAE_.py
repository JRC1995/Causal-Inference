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
import random
# https://github.com/pytorch/examples/blob/master/vae/main.py


class Causal_BVAE(nn.Module):
    def __init__(self):
        super(Causal_BVAE, self).__init__()

        self.z_dim = 20
        self.endo_vars = 20
        self.hidden_dim = 20
        self.classes_num = 1
        self.C = 2

        self.zero = torch.tensor(0.0).cuda()

        self.Encoder = ConvEncoder(self.C, 2*self.z_dim)
        self.mu = nn.Linear(2*self.z_dim, self.z_dim)
        self.logvar = nn.Linear(2*self.z_dim, self.z_dim)
        self.CM = nn.Linear(self.z_dim, self.endo_vars)

        self.alpha = 1  # nn.Parameter(torch.ones(1))
        self.beta = 0  # nn.Parameter(torch.zeros(1))

        self.Decoder = ConvDecoder(self.C, self.endo_vars)

        self.noisy_classifier = nn.Sequential(nn.Linear(self.endo_vars, self.hidden_dim),
                                              nn.ReLU(True),
                                              nn.Linear(self.hidden_dim, self.classes_num))  # 1 == class_num

        self.classifier = nn.Sequential(nn.Linear(self.endo_vars, self.hidden_dim),
                                        nn.ReLU(True),
                                        nn.Linear(self.hidden_dim, self.classes_num))  # 1 == class_num

    def encode(self, x):
        N, C, H, W = x.size()
        h0, h = self.Encoder(x)
        h = F.relu(h)
        h = h.view(N, 2*self.z_dim)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def variable_initialize(self, z):
        v = F.relu(self.CM(z))
        return v

    def IM(self, v, v_):
        N, _ = v.size()

        v_dos = []

        for id in range(self.endo_vars):

            v_do = v.clone()
            v_do[:, id] = v_[:, id]
            v_dos.append(v_do.view(N, 1, self.endo_vars))

        v_dos = torch.cat(v_dos, dim=1)
        v_dos = v_dos.view(self.endo_vars*N, self.endo_vars)

        y = self.noisy_classifier(v.detach())
        y = y.view(N, 1, self.classes_num)
        y = torch.repeat_interleave(y, self.endo_vars, dim=1)

        y_ = self.noisy_classifier(v_dos.detach())
        y_ = y_.view(N, self.endo_vars, self.classes_num)

        imap = torch.abs(y -
                         y_).norm(dim=-1)

        # print(imap.size())

        return imap

    def decode(self, v):
        N, D = v.size()
        v = v.view(N, D, 1, 1)
        h = self.Decoder(v)
        return torch.sigmoid(h)

    def forward(self, x, x_=None, imap=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        v = self.variable_initialize(z)

        if self.training:
            if x_ is None:
                print("Enter x_!")
            else:
                mu_, logvar_ = self.encode(x_)
                z_ = self.reparameterize(mu_, logvar_)
                v_ = self.variable_initialize(z_)
                imap = self.IM(v, v_).detach()
                transformed_map = torch.sigmoid(
                    self.alpha*(imap-imap.mean(dim=-1).view(-1, 1))+self.beta)
        else:
            if imap is None:
                print("Enter influence map (imap)!")
            else:
                imap = imap.view(1, self.endo_vars).detach()
                transformed_map = torch.sigmoid(
                    self.alpha*(imap-imap.mean(dim=-1).view(-1, 1))+self.beta)

        imap_mu = torch.mean(imap, dim=0).view(1, self.endo_vars)
        imap_var = torch.mean(torch.mean((imap-imap_mu).pow(2), dim=0))

        noisy_y = self.noisy_classifier(v)
        y = self.classifier(v*transformed_map)
        recon_x = self.decode(v)

        return imap.sum(dim=0), imap_var, y.view(-1), noisy_y.view(-1), recon_x, mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch\
def ELBO(recon_x, x, mu, logvar, beta=4, epoch=0, iter=0, model='vanilla', save_image_decision=True):

    if save_image_decision:

        pass
        """

        print(recon_x.size())

        save_image(recon_x,
                   'results/'+model+'_sample_' + str(epoch) + '_' + str(iter)+'.png')
        """

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
    return F.binary_cross_entropy_with_logits(logits.view(-1), y.float(), reduction='mean')


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
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()
