# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""

import numpy as np
import matplotlib.pyplot as plt


a = 0 
b = 1


def generate_input():    
    tau = 0.77
    mu = 0.4
    sigma = 0.1
    
    n = 100000
    n_n = int(tau * n)
    n_u = n - n_n
    
    x_n = np.random.normal(mu, sigma, size=n_n)
    x_u = np.random.uniform(a, b, size=n_u)
    x = np.concatenate((x_n, x_u))
    
    return x


def t(x, tau, mu, sigma2):
    "T_ij = tau_j * p_j(x_i) / sum(tau_j * p_j(x_i))"
    
    "p(x_i, z = 'n') = p_n(x_i) * tau"
    p_n_tau_n = (tau 
           / np.sqrt(2*np.pi * sigma2)
           * np.exp(-0.5 * (x - mu)**2 / sigma2))
    
    "p(x_i, z = 'u')"
    idx = (x > a) & (x < b)
    p_u_tau_u = np.zeros_like(x)
    p_u_tau_u[idx] = (1 - tau) / (b - a)
    
    t_n = p_n_tau_n / (p_n_tau_n + p_u_tau_u)
    t_u = p_u_tau_u / (p_n_tau_n + p_u_tau_u)
    
    return t_n, t_u


def theta(x, old):
    "old - $theta^{(t)}$ - (tau, mu, sigma2)"
    t_n, t_u = t(x, *old)
    
    "tau = sum(T_in) / N"
    tau = np.sum(t_n) / x.size
    
    "mu = sum(x_i * T_in) / sum(T_in)"
    mu = np.sum(x * t_n) / np.sum(t_n)
    
    sigma2 = np.sum((x - mu)**2 * t_n) / np.sum(t_n)
    
    return tau, mu, sigma2


def main():
    x = generate_input()
    plt.hist(x, 100)
    tau0 = 0.5
    mu0 = 0.0
    sigma0 = 0.5
    sigma20 = sigma0 ** 2
    
    th = (tau0, mu0, sigma20)
    
    for _ in range(10):
        th = theta(x, th)
        tau, mu, sigma2 = th
        print(
          'tau={:.3f}, mu = {:.3f}, sigma={:.3f}'.format(
            tau, mu, np.sqrt(sigma2))
        )
    
    
if __name__ == '__main__':
    main()
