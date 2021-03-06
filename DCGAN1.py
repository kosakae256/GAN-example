# -*- coding: utf-8 -*-
"""study1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vWtJg63W3hp96n4RGxV0FugA8jR6Ymbi
"""

#GAN
#generatorをresidual networkで構築する例と提案

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout,Reshape,Add
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam


import matplotlib.pyplot as plt

import sys

import numpy as np




class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # 2値分類のための識別モデルを作成
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_dim))
        img = self.generator(z)

        # discriminatorのパラメータ固定
        self.discriminator.trainable = False
        valid = self.discriminator(img)

        self.combined = Model(z, valid)#並列結合
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        k=1.5
        filter=16

        input = Input(shape=(self.latent_dim))
        x = Dense(filter*28*28)(input)
        x = Reshape((28,28,filter))(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        

        for i in range(0,3):
            x2 = Conv2D(filter,(3,3),padding="same",activation='relu')(x)
            x2 = BatchNormalization(momentum=0.8)(x2)
            x2 = LeakyReLU(alpha=0.2)(x2)

            x2 = Conv2D(filter,(3,3),padding="same",activation='relu')(x2)
            x2 = BatchNormalization(momentum=0.8)(x2)
            x = Add()([x2,x])
            x = LeakyReLU(alpha=0.2)(x)
            filter*=k
            x = Dense(filter)(x)

        x = Dense(1)(x)

        output = Activation('tanh')(x)
        model = Model(input,output)

        model.summary()
        return model

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=500):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # noiseからバッチとなる画像をgenerateします。これにはgeneraterを使うよ
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)



            # エポック数表示
            if epoch % save_interval == 0:
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % 500 == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=100000, batch_size=64, save_interval=50)
