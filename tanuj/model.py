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
import cv2
import time

start = time.time()

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

print('Loading data')
# Load data from pickle files
X, y, y_names = deep_utils.load_pickle_files(r"X.p", r"y.p", r"y_names.p")

print('Splitting data')
## Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed, shuffle=True)

# Clear variables for memory
X = None
y = None

print('Reshaping data')
## reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype(np.uint8)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype(np.uint8)

print('Capturing data')
for _ in range(10):
    i = np.random.randint(0, 1000)
    cv2.imwrite(f"test/test{i}.jpg", cv2.cvtColor(X_test[i], cv2.COLOR_RGB2BGR))

print('Normalizing data')
# normalize inputs from 0-255 to 0-1
X_train = np.divide(X_train, 255)
X_test = np.divide(X_test, 255)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_train.shape[1]

def vgg16():
    # Instantiate an empty model
    model = Sequential([
    Convolution2D(64, (3, 3), input_shape=(32, 32, 3), padding='same', activation ='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Convolution2D(128, (3, 3), activation='relu', padding ='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Convolution2D(256, (3, 3), activation='relu', padding ='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Convolution2D(512, (3, 3), activation='relu', padding ='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Convolution2D(512, (3, 3), activation='relu', padding ='same', ),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    Dense(num_classes, activation='softmax')
    ])

    return model

def larger_model():
    '''Define network model'''
    model = Sequential()
    model.add(
        Convolution2D(30, 5, 5, border_mode='valid', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]),
                      activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(15, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', init='he_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', init='he_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu', init='he_normal'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu', init='he_normal'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

print('Building model')
# build the model
model = larger_model()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Fitting model')
# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=2)

finish = time.time()
elapsed = finish - start
print('Runtime :' + str(elapsed) + ' seconds')

deep_utils.plot_accuracy(history)
deep_utils.plot_loss(history)

print(model.summary())

deep_utils.save_model(model, serialize_type='yaml', model_name='facial_recognition_large_cnn_model')
