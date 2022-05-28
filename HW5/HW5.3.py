import scipy.io
import numpy as np
from matplotlib import pyplot as plt

data = scipy.io.loadmat("mnist_49_3000.mat")
x = np.array(data["x"])
y = np.array(data["y"][0])

index = 0 #change the index to show different images
image = x[:,index].reshape(28,28)
plt.imshow(image, interpolation="nearest")
plt.show()