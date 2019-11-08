# Craig Miller
# cmiller@wpi.edu
# Deep Learning for Advanced Robot Perception
#
# Larger CNN for the MNIST Dataset

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from deep_utils import save_model, load_pickle_data

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#Load data from pickle files
X,y,y_names=load_pickle_files('X.p', 'y.p', 'y_names.p')
#Define image size
size=(np.shape(X)[1],np.shape(X)[2])

#TODO: Split data into train/test
#TODO: Add plotting functions
#TODO: Prototype with VOC data

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype('float32')
X_test = X_test.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype('float32')
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
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(size[0], size[1], 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=2)

print(model.summary())
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

save_model(model,serialize_type='yaml',model_name='facial_recognition_large_cnn_model')