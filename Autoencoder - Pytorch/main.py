import numpy as np
import torch
from utils.data import load_data
from utils.model import Autoencoder
from utils.visualize import show_imgs
import os

cwd = os.getcwd()

#loading the data
(x_train, _ ), (x_test, _ ) = load_data()

#expanding dimension to make it of shape (1,28,28)
x_train = np.expand_dims(x_train, axis=1)/255
x_test = np.expand_dims(x_test, axis=1)/255

print("Shape of X_train", x_train.shape)
show_imgs(x_train)
Model = Autoencoder("_model1", x_train, x_test, 100, 64)
Model.train()
PATH = cwd+"/model/autoencoder_weights_model.pt"
torch.save(Model.model.state_dict(), PATH)