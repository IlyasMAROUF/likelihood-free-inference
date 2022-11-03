import numpy as np
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Tanh, Softmax


class MDN(nn.Module):
    def __init__(self, input_size, n_hidden, n_gaussians):
        super().__init__()
        self.hidden_layer = Sequential(
            Linear(input_size, n_hidden),
            Tanh(),
        )
        self.pi_layer = Sequential(
            Linear(n_hidden, n_gaussians),
            Softmax(dim=0)
        )
        self.mu_layer = Linear(n_hidden, n_gaussians)
        self.sigma_layer = Linear(n_hidden, n_gaussians)


    def mog_cdf(self, x, pi, mu, sigma):
        normalization = 1 / np.sqrt(2 * np.pi)
        prob = - 1 / 2 * ((x.expand_as(mu) - mu) / sigma) ** 2
        prob = torch.exp(prob) / sigma * normalization
        prob = prob * pi
        prob = torch.sum(prob, dim=1)
        return prob


    def forward(self, x, y):
        features = self.hidden_layer(x)
        pi = self.pi_layer(features)
        mu = self.mu_layer(features)
        sigma = torch.exp(self.sigma_layer(features))
        probs = self.mog_cdf(y, pi, mu, sigma) # check dimensions
        return probs


    def mdn_loss(self, probs):
        nll = -torch.log(torch.sum(probs, dim=1))
        return torch.mean(nll)