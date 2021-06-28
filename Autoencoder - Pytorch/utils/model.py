import torch
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from utils.visualize import show_nimgs
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),  
            nn.Conv2d(16, 8, 3, padding=1),  
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 8, 3, padding=1),  
            nn.ReLU(True),  
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 8, 3, padding=1), 
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 16, 3, padding=1),  
            nn.ReLU(True),
            nn.Conv2d(16, 1, 3, padding=1),  
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Autoencoder:
    def __init__(self, name, X_train, X_test, epochs, batch_size):
        self.name = name
        self.X_train = X_train
        self.X_test = X_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.model=Model()
    
    def train(self):
        loss_fn=torch.nn.MSELoss(reduction='sum')
        optimizer=torch.optim.Adam(self.model.parameters(),lr=1e-3)
        for epoch in range(self.epochs):
            sum_loss_train=0
            sum_loss_test=0
            for i in tqdm(range(self.X_train.shape[0]//self.batch_size)):
                img=self.X_train[i*self.batch_size:(i+1)*self.batch_size]
                t_img=torch.from_numpy(img).float()
                optimizer.zero_grad()
                output=self.model(t_img)
                loss=loss_fn(output,t_img)
                loss.backward()
                optimizer.step()
                sum_loss_train+=loss.item()
            sum_loss_train=sum_loss_train/(self.X_train.shape[0]//self.batch_size)

            for i in tqdm(range(self.X_test.shape[0]//self.batch_size)):
                img=self.X_test[i*self.batch_size:(i+1)*self.batch_size]
                t_img=torch.from_numpy(img).float()
                output=self.model(t_img)
                loss=loss_fn(output,t_img)
                sum_loss_test+=loss.item()
            sum_loss_test=sum_loss_test/(self.X_test.shape[0]//self.batch_size)
            print("Epoch number",epoch)  
            print("Training loss:",sum_loss_train)
            print("Validation loss:",sum_loss_test) 
            show_nimgs(self.X_train,self.model,epoch)


    