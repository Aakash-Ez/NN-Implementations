from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import pandas as pd
import glob
import os

cwd = os.getcwd()
encoder = load_model(cwd+r"\model\encoder.h5")

fileList = []
feList = []

for file in glob.glob(cwd+r"\data\*.png"):
    filename = file.split("\\")[-1]
    img = load_img(file, color_mode="grayscale")
    img = np.array(img)
    img = np.reshape(img,(1,28,28,1))
    fe = encoder(img)[0].numpy().tolist()
    fileList.append(filename)
    feList.append(fe/np.linalg.norm(fe))

df = pd.DataFrame({"File Name":fileList, "Features":feList})
df.to_json(cwd+r"\features.json")