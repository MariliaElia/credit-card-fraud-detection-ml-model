import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

class GAN:
    def __init__(self, input_dim, generator_output_dim, num_samples, epochs, batch_size):
        self.input_dim = input_dim
        self.generator_output_dim = generator_output_dim
        self.num_samples = num_samples
        self.epochs = epochs
        self.batch_size = batch_size

    def build_discriminator(self):
        discriminator = keras.Sequential(
            [
                layers.Dense(64, input_dim=self.generator_output_dim, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
     
        plot_model(discriminator, 'discriminator.jpg', show_shapes=True,show_dtype=True)
        
        self.discriminator = discriminator

        return discriminator

    def build_generator(self):
        generator = keras.Sequential(
            [
                layers.Dense(64, input_dim=self.input_dim, activation="relu"),
                layers.Dense(64, activation="relu"),
                layers.Dense(self.generator_output_dim, activation="sigmoid"),
            ]
        )

        plot_model(generator, 'generator.jpg', show_shapes=True,show_dtype=True)
        self.generator = generator
    
    def build_gan(self):
        self.discriminator.trainable = False
        gan = keras.Sequential()

        gan.add(self.generator)
        gan.add(self.discriminator)

        self.gan = gan 
        return gan
    
    def train(self, X_train_GAN, y_train_GAN, epochs, batch_size):
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

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" %(epoch + 1, d_loss[0], 100 * d_loss[1], g_loss))

    def generate_samples(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.input_dim))
        synthetic_samples = self.generator.predict(noise)

        synthetic_X = self.ss.inverse_transform(synthetic_samples)
        synthetic_y = np.ones(num_samples)
        return synthetic_X, synthetic_y

    def fit_sample(self, X_train, y_train):
        discriminator = self.build_discriminator()
        # compile discriminator architecture
        discriminator.compile(
                    loss="binary_crossentropy",
                    optimizer=Adam(learning_rate=0.0001, beta_1=0.5),
                    metrics=["accuracy"],
                )

        self.build_generator()

        # make weights in the discriminator not trainable
        discriminator.trainable = False

        cgan = self.build_gan()
        cgan.compile(loss='kld', optimizer=Adam(lr=0.0001, beta_1=0.5))

        self.ss = StandardScaler()
        Xtrain_1 = X_train[y_train == 1]
        X_train_GAN = pd.DataFrame(self.ss.fit_transform(Xtrain_1[:]), index=Xtrain_1.index)
        y_train_GAN = y_train.loc[X_train_GAN.index]

        self.train(X_train_GAN, y_train_GAN, self.epochs, self.batch_size)

        synthetic_X, synthetic_y = self.generate_samples(self.num_samples)

        X_new = np.concatenate((X_train, synthetic_X))
        y_new = np.concatenate((y_train, synthetic_y))

        return(X_new, y_new)



