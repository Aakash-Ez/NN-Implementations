import matplotlib.pyplot as plt
import random
import os
import numpy as np 
cwd = os.getcwd()
size = 28
def show_imgs(X):
  sample = random.sample(range(0, X.shape[0]), 10)
  plt.figure(figsize=(20, 4))
  for i in range(10):
    ax = plt.subplot(1, 10, i + 1)
    plt.imshow(X[sample[i]].reshape(size, size))
  plt.savefig("input_images.png")

def show_nimgs(X, model, epoch):
  sample = random.sample(range(0, X.shape[0]), 10)
  plt.figure(figsize=(20, 4))
  for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    X_true = np.expand_dims(X[sample[i]],axis=0)
    X_pred = model.predict(X_true)
    plt.imshow(X_true.reshape(size, size))
    ax = plt.subplot(2, 10, i + 1 + 10)
    plt.imshow(X_pred.reshape(size, size))
  plt.savefig(cwd+"/output_imgs/epoch-"+str(epoch)+".png")

def compare_imgs(X,Y, model):
  sample = random.sample(range(0, X.shape[0]), 10)
  plt.figure(figsize=(30, 4))
  for i in range(10):
    ax = plt.subplot(3, 10, i + 1)
    X_true = np.expand_dims(X[sample[i]],axis=0)
    X_pred = model.predict(X_true)
    Y_img = np.expand_dims(Y[sample[i]],axis=0)
    plt.imshow(X_true.reshape(size, size))
    ax = plt.subplot(3, 10, i + 1 + 10)
    plt.imshow(X_pred.reshape(size, size))
    ax = plt.subplot(3, 10, i + 1 + 20)
    plt.imshow(Y_img.reshape(size, size))
  plt.savefig("output_img.png")