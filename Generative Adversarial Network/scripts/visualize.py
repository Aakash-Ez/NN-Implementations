import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import random
import os
import numpy as np 
cwd = os.getcwd()
size = 28
channels = 1
def show_imgs(X):
  sample = random.sample(range(0, X.shape[0]), 10)
  plt.figure(figsize=(20, 4))
  for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    if channels > 1:
      plt.imshow(X[sample[i]].reshape(size, size, channels), cmap='gray')
    else:
      plt.imshow(X[sample[i]].reshape(size, size), cmap='gray')
  plt.savefig("input_images.png")

def save_fig(x_fake, epoch):
    for i in range(x_fake.shape[0]):
      if channels > 1:
        e = np.reshape(x_fake[i],(size, size, channels))
      else:
        e = np.reshape(x_fake[i],(size, size))
      plt.subplot(6, 6, 1 + i)
      plt.axis('off')

      plt.imshow(e, cmap='gray')

    filename = 'generated_plot_e%03d.png' % (epoch+1)
    path = cwd+"/output_imgs/"+filename
    plt.savefig(path)
    plt.close()