import numpy as np
from skimage.util import random_noise
def add_noise(image):
    output = np.zeros(image.shape)
    for i in range(image.shape[0]):
      output[i] = random_noise(image[i], mode="gaussian")
    return output