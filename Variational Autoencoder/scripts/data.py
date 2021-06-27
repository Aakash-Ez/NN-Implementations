import pandas as pd
import numpy as np
from tqdm import tqdm

def load_data(path):
    df = pd.read_csv(path)
    Y = df['age'].to_numpy()
    Y = np.reshape(Y, (Y.shape[0], 1))
    X = np.zeros((len(df),48,48,1))
    for index in tqdm(range(len(df))):
        row = df.loc[index]
        arr = [int(i) for i in row['pixels'].split(" ")]
        arr = np.array(arr)
        arr = np.reshape(arr,(48,48,1))
        X[index] = arr/255
    return X, Y