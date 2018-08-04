import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation
from keras.datasets import mnist
import gzip
import sys
import _pickle as cPickle

f = gzip.open('mnist.pkl.gz', 'rb')
if sys.version_info < (3,):
    data = cPickle.load(f)
else:
    data = cPickle.load(f, encoding='bytes')
f.close()
(x_train, y_train), (x_test, y_test) = data

#Load The dataset
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Easier to process values between 0 and 1
X_train = x_train/255
X_test = x_test/255 

#Information about the dataset

print("Number of training examples: ", X_train.shape[0])
print("Number of Classes: ", len(np.unique(y_train)))
print("Shape of an image: ", X_train.shape[1:])

classes, count = np.unique(y_train, return_counts=True)
print("The number of occuranc of each class in the dataset = ", dict (zip(classes, count) ) )

print("\n")
print("Displaying some of the images with labels: ")
images_and_labels = list(zip(X_train, y_train))
#for index, (image, label) in enumerate(images_and_labels[12:24]):
 #   plt.subplot(5,4,index+1)
 #   plt.axis('off')
 #   plt.imshow(image, cmap=plt.cm.gray_r, interpolation = 'nearest')
 #   plt.title('label: %i' % label)


K.set_image_data_format('channels_last')
#Preprocessing the dataset
#as per tensorflow convention - shape = (num_samples, rows, columns, channels)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1], X_test.shape[2], 1).astype('float32')
#one hot encoding the output vector
y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)

#Building the Model

model = Sequential()

model.add(Conv2D(40, kernel_size=5, padding="same", input_shape=(28,28,1), activation='relu'))
model.add(Conv2D(50, kernel_size=5, padding="valid", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(70, kernel_size=3, padding="same", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(100, kernel_size=3, padding="valid", activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation("softmax"))

#Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Training the model
model.fit(X_train, y_train, epochs= 1 , batch_size=200, validation_split = 0.2)

accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: ", accuracy[-1])
