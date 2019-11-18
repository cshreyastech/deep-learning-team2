# Brian wElls
# Updated model derived from original by cmiller
# Deep Learning for Advanced Robot Perception
#
# Larger CNN for the HW7-8, Team #2 Facial Recognition Dataset

import numpy as np
import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
os.environ["KERAS_FLOATX"] = "float16"
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2
import deep_utils
# import tensorflow as tf
import time
from glob import glob
from PIL import Image
import random

# Path for dataset
dataset = r"dl_full_dataset_224x224_split"

start = time.time()
# Initialize tensorflow GPU settings
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
# config = tf.ConfigProto(gpu_options=gpu_options)
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


# K.set_floatx('float32')

def read_data(data_folderpath):
    '''Reads full dataset.  Assumes data has been resized.
    Assumes "data_folderpath" contains 'train' and 'test' subfolders 
    which each have subsubfolders corresponding to class names and each 
    containing jpg files for class.'''
    X_train = np.zeros((70513, 224, 224, 3), dtype=np.float16)  # Full set: 88251, Train:70513, Test:17738
    y_train = np.zeros((70513), dtype=np.uint8)
    X_test = np.zeros((17738, 224, 224, 3), dtype=np.float16)
    y_test = np.zeros((17738), dtype=np.uint8)
    y_names = {}
    # Append folderpaths if needed
    if data_folderpath.endswith('\\') == False:
        data_folderpath = str(data_folderpath) + '\\'
    for dataset in ['train', 'test']:
        # Collect all foldernames
        foldernames = glob(data_folderpath + dataset + '\\' + '*\\')
        if dataset == 'train':
            count = 0
            # Define classes from foldernames
            for idx, foldername in enumerate(foldernames):
                # Append folder names to classes
                y_names[idx] = (foldername.split('\\')[-2])
                print('Loading training data: ' + y_names[idx])
                # Build list of filenames
                filelist = glob(foldername + '*')
                for file in filelist:
                    # Represent classes as integers
                    y_train[count] = idx
                    # Load image
                    image = Image.open(file)
                    # store as np.array
                    X_train[count] = np.array(image)
                    image.close()
                    count += 1
        elif dataset == 'test':
            count = 0
            # Define classes from foldernames
            for idx, foldername in enumerate(foldernames):
                print('Loading testing data: ' + y_names[idx])
                # Build list of filenames
                filelist = glob(foldername + '*')
                for file in filelist:
                    # Represent classes as integers
                    y_test[count] = idx
                    # Load image
                    image = Image.open(file)
                    # store as np.array
                    X_test[count] = np.array(image)
                    image.close()
                    count += 1
        else:
            print('Undefined dataset type')

    return X_train, y_train, X_test, y_test, y_names


def simul_shuffle(mat1, mat2):
    idx = np.arange(0, mat1.shape[0])
    random.shuffle(idx)
    mat1 = mat1[idx]
    mat2 = mat2[idx]
    return mat1, mat2


# Load data
X_train, y_train, X_test, y_test, y_names = read_data(dataset)

# Shuffle data
X_train, y_train = simul_shuffle(X_train, y_train)
X_test, y_test = simul_shuffle(X_test, y_test)

print('Reshaping data')
## reshape to be [samples][width][height][pixels]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype(np.float16)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype(np.float16)

print('Normalizing data')
# normalize inputs from 0-255 to 0-1
# X_train=np.divide(X_train,255).astype(np.float16)
# X_test=np.divide(X_test,255).astype(np.float16)
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]


def larger_model():
    '''Define network model'''
    model = Sequential()
    model.add(
        Convolution2D(30, (5, 5), padding='valid', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),
                      activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, (3, 3), activation='relu', kernel_regularizer=l2(0.005)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.005)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.005)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.005)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.005)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def larger_model_new():
    '''Define network model'''
    model = Sequential()
    model.add(
        Convolution2D(32, (5, 5), padding='valid', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),
                      activation='relu', kernel_regularizer=l2(0.005)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.005)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.005)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.005)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.005)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


print('Building model')
# build the model
model = larger_model()
print('Fitting model')
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=200, verbose=2)

finish = time.time()
elapsed = finish - start
print('Runtime :' + str(elapsed) + ' seconds')

deep_utils.plot_accuracy(history)
deep_utils.plot_loss(history)

print(model.summary())

deep_utils.save_model(model, serialize_type='yaml', model_name='facial_recognition_large_cnn_model')
