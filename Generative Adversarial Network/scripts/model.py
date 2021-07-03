from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from scripts.visualize import save_fig

cwd = os.getcwd()

class GAN_Model:
    def __init__(self, name, X_train, epochs, batch_size):
        self.name = name
        self.X_train = X_train
        self.epochs = epochs
        self.batch_size = batch_size
        self.discriminator = self.create_discriminator(X_train[0].shape)
        self.latent_dim = 64
        self.disc_input = 7*7*self.latent_dim
        self.generator = self.create_generator(self.disc_input)
        self.GAN = self.create_GAN(self.generator, self.discriminator)
        self.x_input = np.random.randn(self.latent_dim * 36)
    def create_discriminator(self, in_shape):
        input_layer = layers.Input(shape=in_shape,name="Input_Layer")
        conv1 = layers.Conv2D(64, (3,3), strides=(2,2), name="Conv_1", padding="same")(input_layer)
        conv1 = layers.LeakyReLU(alpha=0.2)(conv1)
        conv1 = layers.Dropout(0.4)(conv1)
        conv2 = layers.Conv2D(64, (3,3), strides=(2,2), name="Conv_2", padding="same")(conv1)
        conv2 = layers.LeakyReLU(alpha=0.2)(conv2)
        conv2 = layers.Dropout(0.4)(conv2)
        flat = layers.Flatten()(conv2)
        out_layer = layers.Dense(1, activation="sigmoid")(flat)
        
        model = Model(input_layer, out_layer)

        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
    def create_generator(self, latent_dim = 64):
        input_layer = layers.Input(shape=64,name="Input_Layer")
        d1 = layers.Dense(latent_dim)(input_layer)
        reshaped = layers.Reshape((7,7,latent_dim//(7*7)))(d1)
        convT1 = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding='same')(reshaped)
        convT1 = layers.LeakyReLU(alpha=0.2)(convT1)
        convT2 = layers.Conv2DTranspose(256, (5,5), strides=(2,2), padding='same')(convT1)
        convT2 = layers.LeakyReLU(alpha=0.2)(convT2)
        convout = layers.Conv2D(1, (7,7), activation="sigmoid", padding="same")(convT2)
        model = Model(input_layer, convout)

        return model
    def create_GAN(self, gen, dis):
        dis.trainable = False

        model = Sequential()
        model.add(gen)
        model.add(dis)

        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model
    def generate_fake_image(self, n, predict=False):
        if not predict:
          x_input = np.random.randn(self.latent_dim * n)
        else:
          x_input = self.x_input
        x_input = x_input.reshape(n, self.latent_dim)
        samples = self.generator.predict(x_input)
        y = np.zeros((n, 1))
        return samples, y
    def generate_real_image(self, n):
        index = np.random.randint(0, self.X_train.shape[0], n)
        samples = self.X_train[index]
        y = np.ones((n, 1))
        return samples, y
    def train(self):
        n_batch = self.X_train.shape[0]//self.batch_size
        half_batch = self.batch_size//2
        for i in range(self.epochs):
            for j in range(n_batch):
                X_real, y_real = self.generate_real_image(half_batch)
                X_fake, y_fake = self.generate_fake_image(half_batch)
                X_train = np.vstack((X_real, X_fake))
                y = np.vstack((y_real, y_fake))
    
                d_loss, _ = self.discriminator.train_on_batch(X_train, y)
    
                x_input = np.random.randn(self.latent_dim * self.batch_size)
                x_input = x_input.reshape(self.batch_size, self.latent_dim)
                y = np.ones((self.batch_size, 1))
                g_loss = self.GAN.train_on_batch(x_input, y)
                if j%20 == 0:
                    print("Epoch:",i+1, str(j+1)+"/"+str(n_batch), "Discriminator Loss:", d_loss, "Generator Loss:", g_loss)
            print("Epoch:",i+1, str(j+1)+"/"+str(n_batch), "Discriminator Loss:", d_loss, "Generator Loss:", g_loss)
            x_fake, _ = self.generate_fake_image(36, True)
            save_fig(x_fake, i)
    def save(self):
        self.generator.save(cwd+"/model/generator"+self.name+".h5")
        self.discriminator.save(cwd+"/model/discriminator"+self.name+".h5")
        self.GAN.save(cwd+"/model/GAN"+self.name+".h5")