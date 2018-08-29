from keras.datasets import mnist
from keras.layers import *
from keras.layers.convolutional import *
from keras.models import Sequential, Model 
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU

import sys

import numpy as np 

class GAN():

	def __init__(self):
		self.image_rows = 28
		self.image_columns = 28
		self.channels = 1
		self.image_shape = (self.image_rows, self.image_columns, self.channels)
		self.latent_dim = 100

		optimizer = Adam(0.0002, 0.5)

		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

		self.generator = self.build_generator()

		z = Input(shape = (self.latent_dim,))
		image = self.generator(z)

		self.discriminator.trainable = False

		validity = self.discriminator(image)

		self.combined = Model(z, validity)
		self.combined.compile(loss = 'binary_crossentropy', optimizer=optimizer)


	def build_generator(self):

		model = Sequential()

		model.add(Dense(256, input_dim=self.latent_dim))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(1024))
		model.add(LeakyReLU(alpha=0.2))
		model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(np.prod(self.image_shape), activation='tanh'))
		model.add(Reshape(self.image_shape))

		model.summary()

		noise = Input(shape=(self.latent_dim,))
		image = model(noise)

		return Model(noise, image)

	def build_discriminator(self):

		model = Sequential()
		model.add(Flatten(input_shape = self.image_shape))
		model.add(Dense(512))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(256))
		model.add(LeakyReLU(alpha=0.2))
		model.add(Dense(1, activation = 'sigmoid'))
		model.summary()

		image = Input(shape=self.image_shape)
		validity = model(image)

		return Model(image, validity)

	def train(self, epochs, batch_size=128, sample_interval=50):


		f = gzip.open('mnist.pkl.gz', 'rb')
		if sys.version_info < (3,):
			data = cPickle.load(f)
		else:
			data = cPickle.load(f, encoding='bytes')
		f.close()

		(X_train, y_train), (x_test, y_test) = data

		X_train = X_train / 127.5 - 1.
		X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
		valid = np.ones((batch_size, 1))
		fake = np.zeros((batch_size, 1))

		for epoch in range(epochs):

            #  Train Discriminator

            # Select a random batch of images
			idx = np.random.randint(0, X_train.shape[0], batch_size)
			imgs = X_train[idx]

			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
			gen_imgs = self.generator.predict(noise)

            # Train the discriminator
			d_loss_real = self.discriminator.train_on_batch(imgs, valid)
			d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #  Train Generator

			noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
			g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
			print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))



if __name__ == '__main__':
	gan = GAN()
	gan.train(epochs = 1000, batch_size = 32, sample_interval = 200)