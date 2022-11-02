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


class MA2Prior:
    def rvs(self):
        theta1 = stats.uniform.rvs(-2, 4, size=None)
        theta2 = stats.uniform.rvs(np.abs(theta1), 1, size=None)
        return [theta1, theta2]

    def cdf(self, theta1, theta2):
        theta1_prob = stats.uniform(-2, 4).cdf(theta1)
        theta2_prob = stats.uniform(np.abs(theta1)-1, 2-np.abs(theta1)).cdf(theta2)
        return theta1_prob * theta2_prob


def autocovariance(ts, k):
    ts_mean = np.mean(ts)
    autocov = sum([(ts[i+k] - ts_mean) * (ts[i] - ts_mean) for i in range(len(ts)-k)])
    autocov /= len(ts) - 1
    return autocov