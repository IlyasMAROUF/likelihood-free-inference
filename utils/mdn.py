import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, Tanh, Softmax


class MDN(nn.Module):
    def __init__(self, input_size, n_hidden, n_gaussians, gaussian_dim):
        super().__init__()
        self.n_gaussians = n_gaussians
        self.gaussian_dim = gaussian_dim

        self.hidden_layer = Sequential(
            Linear(input_size, n_hidden),
            Tanh(),
        )
        self.pi_layer = Sequential(
            Linear(n_hidden, n_gaussians),
            Softmax(dim=0)
        )
        self.mu_layer = Linear(n_hidden, n_gaussians * gaussian_dim)
        self.sigma_layer = Linear(n_hidden, n_gaussians * gaussian_dim ** 2)


    def mog_cdf(self, y, pi, mu, sigma):
        y = y.expand_as(mu)
        normalization = 1 / torch.sqrt(torch.linalg.det(sigma) * torch.pi ** self.gaussian_dim)
        prob = - 1/2 * torch.dot(torch.matmul(y-mu, torch.linalg.inv(sigma)), y-mu)
        prob = torch.exp(prob) * normalization
        prob = prob * pi
        prob = torch.sum(prob, dim=1)
        return prob


    def forward(self, x, y):
        features = self.hidden_layer(x)
        pi = self.pi_layer(features)
        mu = self.mu_layer(features)
        sigma = self.sigma_layer(features)

        if self.gaussian_dim == 1:
            sigma = torch.exp(sigma)
        else:
            mu = mu.reshape((self.n_gaussians, self.gaussian_dim))
            sigma = sigma.reshape((self.n_gaussians, self.gaussian_dim ** 2))
            assert all(torch.linalg.det(sigma) != 0), "Not all sigmas are invertible"

        probs = self.mog_cdf(y, pi, mu, sigma)
        return probs


    def mdn_loss(self, probs):
        nll = -torch.log(torch.sum(probs, dim=1))
        return torch.mean(nll)