import numpy as np
from scipy import stats


class MAProcess:
    def __init__(self, *theta):
        self.q = len(theta)
        self.theta = list(theta)

    def rvs(self, size):
        u = np.random.normal(0, 1, size+self.q)
        z = [u[i+self.q] + sum(u[i:i+self.q-1] * self.theta) for i in range(size)]
        return z


class Theta1Prior(stats.rv_continuous):
    def _pdf(self, theta1):
        if -2 < theta1 and theta1 < 2:
            return (2 - np.abs(theta1)) / 4
        else:
            return 0


class MA2Prior:
    def __init__(self):
        self.theta1prior = Theta1Prior()

    def rvs(self, size=None):
        theta1 = self.theta1prior.rvs(size=size)
        theta2 = stats.uniform.rvs(np.abs(theta1), 2-np.abs(theta1), size=size)
        return np.c[theta1, theta2]

    def cdf(self, theta1, theta2):
        return 1/4


def autocovariance(ts, k):
    ts_mean = np.mean(ts)
    autocov = sum([(ts[i+k] - ts_mean) * (ts[i] - ts_mean) for i in range(len(ts)-k)])
    autocov /= len(ts) - 1
    return autocov