from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from scipy.spatial.distance import cosine, euclidean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

cwd = os.getcwd()
encoder = load_model(cwd+r"\model\encoder.h5")

def show_imgs(query,X):
    plt.figure(figsize=(20, 4))
    ax = plt.subplot(2, 5, 3)
    plt.imshow(query.reshape(28,28))
    plt.title("Query Image")
    for i in range(len(X)):
        ax = plt.subplot(2, 5, i + 6)
        img = np.array(load_img(cwd+"\\data\\"+X[i], color_mode="grayscale"))
        plt.imshow(img.reshape(28, 28))
        plt.title("Image - "+str(i+1))
    plt.savefig("predicted_images.png")

img = load_img("query.png", color_mode="grayscale")
img = np.array(img)
img = np.reshape(img,(1,28,28,1))

fe = encoder(img)[0].numpy().tolist()
fe = fe/np.linalg.norm(fe)

df = pd.read_json("features.json")
difference = []

for index, row in df.iterrows():
    diff = euclidean(row["Features"], fe)
    difference.append(diff)

df['Difference'] = difference
df.sort_values("Difference",inplace=True)
df = df.head(5)
fileList = df["File Name"].to_list()
show_imgs(img, fileList)