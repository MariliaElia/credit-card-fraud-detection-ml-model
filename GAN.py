import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.utils import plot_model
from keras import backend as K

class GAN:
    def __init__(self, generator_output_dim, discriminator_input_dim, noise_dim, num_samples, epochs, batch_size, dropout):
        self.generator_output_dim = generator_output_dim
        self.discriminator_input_dim = discriminator_input_dim
        self.noise_dim = noise_dim
        self.num_samples = num_samples
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout

    def build_discriminator(self):
        discriminator = keras.Sequential([
            layers.Dense(20, input_dim=self.discriminator_input_dim,
                         kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(self.dropout),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(10, kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(self.dropout),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(1, activation='sigmoid')
        ])
     
        plot_model(discriminator, 'discriminator.jpg', show_shapes=True,show_dtype=True)
        discriminator.compile(
                    loss="binary_crossentropy",
                    optimizer=Adamax(lr=0.00002, beta_1=0.5),
                    metrics=["accuracy"],
                )
        
        self.discriminator = discriminator

    def build_generator(self):
        generator = keras.Sequential([
            layers.Dense(128, input_dim=self.noise_dim, activation='relu', kernel_initializer='glorot_uniform',
                         kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(self.dropout),
            layers.Dense(64, activation='relu', kernel_initializer='glorot_uniform',
                         kernel_regularizer=regularizers.l2(0.01)),
            layers.Dropout(self.dropout),
            layers.Dense(self.generator_output_dim, activation='tanh')
        ])

        plot_model(generator, 'generator.jpg', show_shapes=True,show_dtype=True)
        self.generator = generator
    
    def build_gan(self):
        self.discriminator.trainable = False
        gan = keras.Sequential()
        gan.add(self.generator)
        gan.add(self.discriminator)
       
        gan.compile(loss=self.generator_loss, optimizer=Adamax(lr=0.00002, beta_1=0.5))
        gan.summary()

        self.gan = gan

    # create a line plot of loss for the gan and save to file
    def plot_history(self, d_real_hist, d_fake_hist, g_hist, a_real_hist, a_fake_hist):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.plot(d_real_hist, label='d-real')
        plt.plot(d_fake_hist, label='d-fake')
        plt.plot(g_hist, label='gen')
        plt.legend()
        # plot discriminator accuracy
        plt.subplot(2, 1, 2)
        plt.plot(a_real_hist, label='acc-real')
        plt.plot(a_fake_hist, label='acc-fake')
        plt.legend()

        plt.show()
    
    def train(self, X_train_GAN, y_train_GAN, epochs, batch_size):
        real = np.ones(batch_size)
        fake = np.zeros(batch_size)
        # prepare lists for storing stats each iteration
        d_real_hist, d_fake_hist, g_hist, a_real_hist, a_fake_hist = list(), list(), list(), list(), list()

        for epoch in range(epochs):
            # Select random samples from the fraudulent training data
            indices = np.random.choice(y_train_GAN, batch_size, replace=False)
            real_samples = X_train_GAN.iloc[indices]
          
            latent_points = np.random.normal(0, 1, (batch_size, self.noise_dim))
            generated_samples = self.generator.predict(latent_points)

            # pass real and generated samples to the discriminator and train on them
            d_loss_real, d_acc_real = self.discriminator.train_on_batch(real_samples, real)
            d_loss_fake, d_acc_fake = self.discriminator.train_on_batch(generated_samples, fake)
            
            d_loss = 0.5*(d_loss_real + d_loss_fake)
            d_acc = 0.5*(d_acc_real + d_acc_fake)

            noise = np.random.normal(0, 1, (batch_size, self.noise_dim))
            g_loss = self.gan.train_on_batch(noise, real)

            # record history
            d_real_hist.append(d_loss_real)
            d_fake_hist.append(d_loss_fake)
            g_hist.append(g_loss)
            a_real_hist.append(d_acc_real)
            a_fake_hist.append(d_acc_fake)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %(epoch + 1, d_loss, 100 * d_acc, g_loss))
        
        self.plot_history(d_real_hist, d_fake_hist, g_hist, a_real_hist, a_fake_hist)

    def generate_samples(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.noise_dim))
        synthetic_samples = self.generator.predict(noise)

        synthetic_X = self.ss.inverse_transform(synthetic_samples)
        synthetic_y = np.ones(num_samples)
        return synthetic_X, synthetic_y
    
    # Total Generator Loss
    def generator_loss(self,y_true, y_pred):
        bce_loss = keras.losses.binary_crossentropy(y_true, y_pred)
        kl_loss = keras.losses.kld(y_true, y_pred)
        total_loss = K.mean(bce_loss + kl_loss)
        return total_loss

    def fit_sample(self, X_train, y_train):
        self.build_discriminator()
        self.build_generator()
        self.build_gan()
        
        self.ss = StandardScaler()
        Xtrain_1 = X_train[y_train == 1]
        X_train_GAN = pd.DataFrame(self.ss.fit_transform(Xtrain_1[:]), index=Xtrain_1.index)
        y_train_GAN = y_train.loc[X_train_GAN.index]

        self.train(X_train_GAN, y_train_GAN, self.epochs, self.batch_size)

        synthetic_X, synthetic_y = self.generate_samples(self.num_samples)

        X_new = np.concatenate((X_train, synthetic_X))
        y_new = np.concatenate((y_train, synthetic_y))

        return(X_new, y_new)




