import matplotlib.pyplot as plt
from scripts.data import load_data
from tensorflow.keras.models import load_model
from scripts.model import VAE, Sampling
from tensorflow.keras.utils import get_custom_objects
import numpy as np
import os

def plot_label_clusters(vae, data, labels):
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig("latent_space.png")
    plt.show()

cwd = os.getcwd()

data_dir = cwd+"/data/age_gender.csv"
X, Y = load_data(data_dir)

X = np.expand_dims(X, -1)

get_custom_objects()['Sampling'] = Sampling

encoder = load_model("encoder.h5")
decoder = load_model("decoder.h5")
vae = VAE(encoder, decoder)
plot_label_clusters(vae, X, Y)