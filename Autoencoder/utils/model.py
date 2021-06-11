from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import Callback
import os
from utils.visualize import show_nimgs

cwd = os.getcwd()

class epochImg(Callback):
  def __init__(self, data):
        self.X = data
  def on_epoch_end(self, epoch, logs = {}):
        show_nimgs(self.X,self.model,epoch)

class Autoencoder:
    def __init__(self, name, X_train, X_test, epochs, batch_size):
        self.name = name
        self.X_train = X_train
        self.X_test = X_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.encoder, self.autoencoder = self.create_model()
        self.decoder = self.create_decoder(128,8,15)
        self.history = ""
        self.callbacks = [epochImg(X_test)]
    def create_model(self):
        input_layer = layers.Input(shape=self.X_train[0].shape,name="Input_Layer")
        conv1 = layers.Conv2D(16, (3,3), name="Conv_1", activation="relu", padding="same")(input_layer)
        conv1 = layers.MaxPooling2D((2,2), padding="same", name="Maxpooling1")(conv1)
        conv2 = layers.Conv2D(8, (3,3), name="Conv_2", activation="relu", padding="same")(conv1)
        conv2 = layers.MaxPooling2D((2,2), padding="same", name="Maxpooling2")(conv2)
        conv3 = layers.Conv2D(8, (3,3), name="Conv_3", activation="relu", padding="same")(conv2)
        conv3 = layers.MaxPooling2D((2,2), padding="same", name="Maxpooling3")(conv3)

        latent_vector = layers.Flatten()(conv3)
        encoded = layers.Reshape((4,4,8))(latent_vector)

        up1 = layers.Conv2D(8, (3, 3), name="Conv1T", activation="relu", padding="same")(encoded)
        up1 = layers.UpSampling2D((2, 2), name="Up1")(up1)
        up2 = layers.Conv2D(8, (3, 3), name="Conv2T", activation="relu", padding="same")(up1)
        up2 = layers.UpSampling2D((2, 2), name="Up2")(up2)
        up3 = layers.Conv2D(16, (3, 3), name="Conv3T", activation="relu")(up2)
        up3 = layers.UpSampling2D((2, 2), name="Up3")(up3)
        output = layers.Conv2D(1, (3, 3), name="Output", activation="sigmoid", padding="same")(up3)
        encoder = Model(input_layer,latent_vector)
        autoencoder = Model(input_layer,output)
        return encoder, autoencoder
    def create_decoder(self, encoding_dim, start_ix, end_ix):
        model = Sequential()
        model.add(layers.Input(shape=(encoding_dim), name="Input"))
        for ix in range(start_ix,end_ix+1):
            curr_layer = self.autoencoder.layers[ix]
            model.add(curr_layer)
        return model
    def train(self):
        self.autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
        self.history = self.autoencoder.fit(self.X_train, self.X_train, epochs=self.epochs, batch_size=self.batch_size, 
        shuffle=True, validation_data=(self.X_test, self.X_test), callbacks=self.callbacks)
    def save(self):
        self.autoencoder.save(cwd+"/model/autoencoder"+self.name+".h5")
        self.encoder.save(cwd+"/model/encoder"+self.name+".h5")
        self.decoder.save(cwd+"/model/decoder"+self.name+".h5")