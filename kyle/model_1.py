# Craig Miller
# cmiller@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Larger CNN for the MNIST Dataset

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D,ZeroPadding2D
from keras.utils import np_utils
import deep_utils
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
#TODO: Why is y_names not printing right?

#Initialize tensorflow GPU settings
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
#config = tf.ConfigProto(gpu_options=gpu_options)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

img_rows,img_cols = 32,32
#Load data from pickle files
X,y,y_names=deep_utils.load_pickle_files('X.p', 'y.p', 'y_names.p')
#print(np.shape(y))
#print(y_names) 

# Load and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

print('Begin reshaping input data...')
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype(np.float16)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype(np.float16)
print('Completed reshaping data.')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def larger_model():
    '''Define network model'''
    model = Sequential()
    model.add(Convolution2D(64, 3, 3, border_mode='valid', input_shape=(img_cols,img_rows, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
    
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', init='glorot_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',init='glorot_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# build the model
model = larger_model()
model.summary()
# Fit the model
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=64, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

deep_utils.plot_accuracy(history)
deep_utils.plot_loss(history)

print(model.summary())
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

deep_utils.save_model(model,serialize_type='yaml',model_name='facial_recognition_large_cnn_model')
#Save history
#pickle.dump(history, open("history.p", "wb"))
