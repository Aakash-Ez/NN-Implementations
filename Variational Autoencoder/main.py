from scripts.model import create_encoder
from scripts.data import load_data
from scripts.model import create_encoder, create_decoder, VAE, epochImg
from tensorflow.keras.optimizers import Adam
from scripts.visualize import show_imgs
import os
import random

cwd = os.getcwd()
data_dir = cwd+"/data/age_gender.csv"
X, Y = load_data(data_dir)
show_imgs(X)
print(X.shape)
encoder = create_encoder(X[0].shape)
decoder = create_decoder()

vae = VAE(encoder, decoder)
vae.compile(optimizer=Adam(0.01))

sample = random.sample(range(0, X.shape[0]), 10)

vae.fit(X, epochs=30, batch_size=64, callbacks=[epochImg(X, sample)])

vae.encoder.save("encoder.h5")
vae.decoder.save("decoder.h5")