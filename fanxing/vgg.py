# Craig Miller
# cmiller@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Larger CNN for the MNIST Dataset

import numpy as np
from keras.engine import  Model
from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Input
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import deep_utils
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
import time
from keras_vggface.vggface import VGGFace

#TODO: Why is y_names not printing right?

start=time.time()
#Initialize tensorflow GPU settings
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.ConfigProto(gpu_options=gpu_options)
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print('Loading data')
#Load data from pickle files
X,y,y_names=deep_utils.load_pickle_files(r"/home/albert/deep-learning-team2/X.p", r"/home/albert/deep-learning-team2/y.p", r"/home/albert/deep-learning-team2/y_names.p")
#print(np.shape(y))

print('Splitting data')
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
#Clear variables for memory
X=None
y=None

print('Reshaping data')
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype(np.uint8)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype(np.uint8)

print('Normalizing data')
# normalize inputs from 0-255 to 0-1
for idx, mat in enumerate(X_train):
    X_train[idx]=X_train[idx]/255
for idx, mat in enumerate(X_test):
    X_test[idx]=X_test[idx]/255
    
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def larger_model():
	'''Define network model'''
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
#	model.add(Convolution2D(15, 3, 3, activation='relu'))
#	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu',init='he_normal'))
	model.add(Dropout(0.2))
	model.add(Dense(128, activation='relu',init='he_normal'))
	model.add(Dropout(0.2))
	model.add(Dense(64, activation='relu',init='he_normal'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
nb_class = 7
hidden_dim = 512

# build the model
vgg_model =  VGGFace(include_top=False, input_shape=(224, 224, 3))

last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
model = Model(vgg_model.input, out)
try:
	model = multi_gpu_model(model)
except:
	pass
# Fit the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=9, batch_size=32, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

finish=time.time()
elapsed=finish-start
print('Runtime :'+str(elapsed)+' seconds')

deep_utils.plot_accuracy(history)
deep_utils.plot_loss(history)

#print(model.summary())
#print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

deep_utils.save_model(model,serialize_type='yaml',model_name='facial_recognition_large_cnn_model')
#Save history
#pickle.dump(history, open("history.p", "wb"))
