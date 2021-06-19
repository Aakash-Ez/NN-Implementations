import numpy as np
from scripts.model import Autoencoder
from scripts.visualize import show_imgs, compare_imgs
from scripts.data import add_noise
from tensorflow.keras.datasets.fashion_mnist import load_data

#loading the data
(y_train, _ ), (y_test, _ ) = load_data()

y_train = np.expand_dims(y_train, axis=-1)/255
y_test = np.expand_dims(y_test, axis=-1)/255

x_train = add_noise(y_train)
x_test = add_noise(y_test)

print("Shape of X_train", x_train.shape)
show_imgs(y_train)
Model = Autoencoder("_model1", x_train, x_test, y_train, y_test, 90, 32)
print(Model.decoder.summary())
Model.autoencoder.load_weights("model/autoencoder_model1.h5")
Model.train()

Model.save()
compare_imgs(x_test, y_test, Model.autoencoder)