import tensorflow as tf

from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
from keras.losses import mse
from keras import backend as K

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

# Visual Libraries
from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

class VAE_oversampling:
    def __init__(self, epochs, hidden_dim,
                 batch_size, latent_dim, original_dim,
                 minority_class_id, 
                 random_state, num_samples_to_generate,
                 optimizer = "adam"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.original_dim = original_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.minority_class_id = minority_class_id
        self.random_state = random_state
        self.num_samples_to_generate = num_samples_to_generate
        self.optimizer = optimizer

        #set random seed
        np.random.seed(random_state)

    #Reparameterization function
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2))
        return z_mean + K.exp(0.5*z_log_var) * epsilon
    
    def display_vae_training_history(self, history):        
        plt.figure(figsize=(6,3))
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val loss')
        plt.ylabel('MSE + KLD')
        plt.xlabel('No. epoch')
        plt.legend(loc="upper left") 
        plt.title("Autoencoder Training History")
        plt.show()
        
    def build_train_vae(self, X_train_AE):
        # Mapping inputs to latent distribution parameters
        inputs = Input(shape=(self.original_dim,))
        h = Dense(self.hidden_dim, activation='relu')(inputs)

        #Latent space layer
        z_mean = Dense(self.latent_dim)(h)
        z_log_sigma = Dense(self.latent_dim)(h)

        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_log_sigma])

        #Mapping these sampled latent points back to reconstructed inputs
        # Create encoder
        encoder = Model(inputs, [z_mean, z_log_sigma, z], name='encoder')

        # Create decoder
        latent_inputs = Input(shape=(self.latent_dim,), name='z_sampling')
        x = Dense(self.hidden_dim, activation='relu')(latent_inputs)
        outputs = Dense(self.original_dim, activation='sigmoid')(x)
        decoder = Model(latent_inputs, outputs, name='decoder')

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='vae_mlp')

        #Caclulate reconstruction from input and output
        reconstruction_loss = mse(inputs, outputs)
        reconstruction_loss *= self.original_dim
            
        #Kullback-liebler divergence loss
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        
        #The total vae loss
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)

        vae.compile(optimizer=self.optimizer)

        history = vae.fit(X_train_AE, X_train_AE, self.batch_size, self.epochs, validation_split=0.1)

        self.display_vae_training_history(history)

        self.encoder = encoder
        self.decoder = decoder

        return vae
    
    def fit_sample(self, Xtrain, ytrain):
        #Number of samples to generate
        num_samples_to_generate = self.num_samples_to_generate
        
        #Scale the data set
        ss = StandardScaler()
        Xtrain_1 = Xtrain[ytrain == self.minority_class_id]
        X_train_AE_scaled = ss.fit_transform(Xtrain_1[:])

        #Pass data set to the build function
        self.build_train_vae(X_train_AE_scaled)
        
        #randomly sample from standard normal
        z_latent_sample = np.random.normal(0, 1,
                                    (num_samples_to_generate,
                                     self.latent_dim))
        
        #Generate the synthetic samples by passing the z sample
        synthetic_samples = self.decoder.predict(z_latent_sample)
            
        synthetic_X = ss.inverse_transform(synthetic_samples)
        synthetic_y = np.ones(num_samples_to_generate)\
            * self.minority_class_id
        
        #Final step, concetenate original observations with synthetic observations
        X_new = np.concatenate((Xtrain, synthetic_X))
        y_new = np.concatenate((ytrain, synthetic_y))
        return(X_new, y_new)