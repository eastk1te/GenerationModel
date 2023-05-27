"""

"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras import optimizers


import numpy as np
import os
import matplotlib.pyplot as plt


class GAN(tf.keras.Model):

    def __init__(self, noise=100, image_shape=(1, 28, 28)):
        super().__init__()

        self.noise_dim = noise
        self.image_shape = image_shape
        self.BATCH_SIZE = 256

        # setting enviroment.
        self.device = tf.device('cuda' if tf.config.list_physical_devices('GPU') else 'cpu')
        if self.device.type == 'cuda':
            print(f'"GPU" is Available')
        else:
            print(f'"CPU" is Available.')

        self.generator = self.Generator()
        self.discriminator = self.Discriminator()

        optimizer_G = optimizers.Adam(2e-4)
        optimizer_D = optimizers.Adam(2e-4)



        # torch.save(generator.state_dict(), 'pytorch_weights_gen.pt')
        # torch.save(discriminator.state_dict(), 'pytorch_weights_dis.pt')

    def dense_block(self, units, input_dim=None, normalize=False, momentum=0.8, alpha=0.2):
        block = Sequential()
        block.add(layers.Dense(units, input_dim=input_dim))

        if normalize:
            block.add(layers.BatchNormalization(momentum=momentum))

        block.add(layers.LeakyReLU(alpha=alpha))
        
        return block

    def Generator(self, noise):
        
        model = Sequential(
            [self.dense_block(128, normalize=True, input_dim=noise),
             self.dense_block(256, normalize=True),
             self.dense_block(512, normalize=True),
             layers.Dense(tf.reduce_prod(noise), activation='tanh'),
             layers.Reshape(self.image_shape)
             ]
        )

        return model

    def Discriminator(self):
        model = Sequential(
            [layers.Flatten(input_shape=self.image_shape),
             self.dense_block(512),
             self.dense_block(256),numpy imageca
             layers.Dense(1, activation='sigmoid')
             ]
        )
        return model

    def cross_entropy(self):
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

        real_output = self.discriminator(images, training=True)
        fake_output = self.discriminator(generated_images, training=True)

        gen_loss = self.generator_loss(fake_output)
        disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.optimizer_G.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.optimizer_D.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))


    def train(self, dataset, num_epochs):
        for epoch in range(num_epochs):
            for i, img in enumerate(dataset):
                self.train_step(img)

if __name__ == '__main__':
    
    # https://www.kaggle.com/competitions/quickdraw-doodle-recognition/data