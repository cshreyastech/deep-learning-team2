# Craig Miller
# cmiller@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Larger CNN for the HW7-8, Team #2 Facial Recognition Dataset

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import deep_utils
from sklearn.model_selection import train_test_split
import tensorflow as tf
import time

start=time.time()
#Initialize tensorflow GPU settings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print('Loading data')
#Load data from pickle files
X,y,y_names=deep_utils.load_pickle_files(r"X.p", r"y.p", r"y_names.p")

print('Splitting data')
## Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed,shuffle=True)

#Clear variables for memory
X=None
y=None

print('Reshaping data')
## reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype(np.uint8)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype(np.uint8)

print('Normalizing data')
# normalize inputs from 0-255 to 0-1
X_train=np.divide(X_train,255)
X_test=np.divide(X_test,255)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes=y_train.shape[1]

def larger_model():
	'''Define network model'''
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu',init='he_normal'))
	model.add(Dropout(0.2))
	model.add(Dense(128, activation='relu',init='he_normal'))
	model.add(Dropout(0.2))
	model.add(Dense(128, activation='relu',init='he_normal'))
	model.add(Dropout(0.2))
	model.add(Dense(64, activation='relu',init='he_normal'))
	model.add(Dense(10, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

print('Building model')
# build the model
model = larger_model()
print('Fitting model')
# Fit the model
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=128, verbose=2)

finish=time.time()
elapsed=finish-start
print('Runtime :'+str(elapsed)+' seconds')

deep_utils.plot_accuracy(history)
deep_utils.plot_loss(history)

print(model.summary())

deep_utils.save_model(model,serialize_type='yaml',model_name='facial_recognition_large_cnn_model')