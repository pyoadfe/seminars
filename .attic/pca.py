# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from PIL import Image
import glob


dim = 10
samples = 100
k = 5
cov = np.diag(np.arange(1, dim+1, dtype=float))
gauss = multivariate_normal(mean=np.zeros(dim),
                            cov=cov)
data = gauss.rvs(size=samples)
print(data.shape)
u, s, v = np.linalg.svd(data)
print(s)
encoder = v[:k]

x = data[0]
delta = v.T @ (v @ x) - x
print(delta)

encoded_x = encoder @ x
x_transcoded = encoder.T @ encoded_x
print('x', x)
print('encoded', encoded_x)
print('delta', x - x_transcoded)


u, s, v = np.linalg.svd(data)
plt.figure()
plt.title('Principal components')
for i in range(3):
    plt.plot(v[i], label=str(i))
plt.legend()


path = r'D:\Downloads\emoji\emoji\*.png'
images = [Image.open(f).convert('L')
          for f in  glob.glob(path)]
shape = images[0].size
samples = [np.array(image, dtype=float).reshape(-1)
           for image in images]
data = np.vstack(samples)
plt.imshow(data[10].reshape(shape), cmap='gray')
mean = data.mean(axis=0)
data -= mean
plt.figure()
u, s, v = np.linalg.svd(data)
for i in range(5):
    plt.figure()
    plt.title(str(i))
    plt.imshow(v[i].reshape(shape), label=str(i),
               cmap='gray')

x = data[10]
encoder = v[:20]
transcoded = encoder.T @ (encoder @ x)
plt.figure()
plt.imshow(transcoded.reshape(shape) + mean.reshape(shape), cmap='gray')



