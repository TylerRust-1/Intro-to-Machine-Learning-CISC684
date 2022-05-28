import csv
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from matplotlib import style
#style.use('fivethirtyeight')
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs

x = np.linspace(-10, 10, 500)
y = np.linspace(-10, 10, 500)
X, Y = np.meshgrid(x,y)

print(X)
print(Y)