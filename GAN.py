import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

class GAN:
    def __init__(self, input_dim, generator_output_dim):
        self.input_dim = input_dim
        self.generator_output_dim = generator_output_dim

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()

        self.gan = self.build_gan()

        opt = Adam(lr=0.00002, beta_1=0.5)
        self.gan.compile(loss='binary_crossentropy', optimizer=opt)

    def build_discriminator(self):
        discriminator = keras.Sequential(
            [
                layers.Dense(64, input_dim=self.generator_output_dim, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
     
        plot_model(discriminator, ',/discriminator.jpg', show_shapes=True,show_dtype=True)
        
        return discriminator

    def build_generator(self):
        generator = keras.Sequential(
            [
                layers.Dense(64, input_dim=self.input_dim, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(self.generator_output_dim, activation="sigmoid"),
            ]
        )

        plot_model(generator, './generator.jpg', show_shapes=True,show_dtype=True)

        return generator
    
    def build_gan(self):
        self.discriminator.trainable = False
        gan = keras.Sequential()

        gan.add(self.generator)
        gan.add(self.discriminator)

        return gan
    
    def train(self, X_train, y_train, epochs, batch_size):
        ss = StandardScaler()
        Xtrain_1 = X_train[y_train == 1]
        X_train_GAN = pd.DataFrame(ss.fit_transform(Xtrain_1[:]), index=Xtrain_1.index)
        y_train_GAN = y_train.loc[X_train_GAN.index]

        print("Prepare data for training:")
        print(X_train_GAN.shape, y_train_GAN.shape)
        print(X_train_GAN.index, y_train_GAN.index)

        real = np.ones(batch_size)
        fake = np.zeros(batch_size)


        for epoch in range(epochs):
            # Select random samples from the fraudulent training data
            indices = np.random.choice(y_train_GAN, batch_size, replace=False)
            real_samples = X_train_GAN.iloc[indices]

            noise = np.random.normal([batch_size, self.input_dim])
            latent_points = np.random.normal(0, 1, (batch_size, self.input_dim))
            generated_samples = self.generator.predict(latent_points)

            # Combine real and fake data
            X = np.concatenate((real_samples, generated_samples))
            y = np.concatenate((real, fake))

            d_loss = self.discriminator.train_on_batch(X, y)

            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            g_loss = self.gan.train_on_batch(noise, real)

            print(f"Epoch {epoch + 1}/{epochs} - D Loss: {d_loss} - G Loss: {g_loss}")

    def generate_samples(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.input_dim))
        synthetic_samples = self.generator.predict(noise)

        X_new = self.ss.inverse_transform(synthetic_samples)
        y_new = np.ones((num_samples, 1))
        return X_new, y_new
