import numpy as np
from scripts.model import GAN_Model
from scripts.visualize import show_imgs
from scripts.data import load_data
from keras.datasets.mnist import load_data

(x_train, _), (x_test, _) = load_data()
x_train = np.concatenate((x_train, x_test), axis=0)
X = np.expand_dims(x_train, axis=-1)
X = X.astype('float32')
X = X/255

show_imgs(X)

print("Shape of X_train", X.shape)
Model = GAN_Model("_model1", X, 100, 256)
print(Model.generator.summary())
Model.train()

Model.save()