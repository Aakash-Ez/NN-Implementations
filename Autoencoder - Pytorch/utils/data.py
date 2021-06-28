import numpy as np
import os

cwd = os.getcwd()
def load_data():
    x_train = np.load(cwd+"/data/k49-train-imgs.npz")['arr_0']
    y_train = np.load(cwd+"/data/k49-train-labels.npz")['arr_0']
    x_test = np.load(cwd+"/data/k49-test-imgs.npz")['arr_0']
    y_test = np.load(cwd+"/data/k49-test-labels.npz")['arr_0']
    return (x_train,y_train), (x_test,y_test)
