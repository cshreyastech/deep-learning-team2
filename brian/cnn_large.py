# Larger CNN for the MNIST Dataset

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

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

#Load data
X,y,y_names=deep_utils.load_pickle_files(r"pickle/X.p", r"pickle/y.p", r"pickle/y_names.p")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3]).astype(np.uint8)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], X_test.shape[3]).astype(np.uint8)

# normalize inputs from 0-255 to 0-1
for idx, mat in enumerate(X_train):
	X_train[idx] = X_train[idx] / 255
for idx, mat in enumerate(X_test):
	X_test[idx] = X_test[idx] / 255

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def lenet5_model():
	# Update to LeNet-5 architecture
	model = Sequential()
	model.add(Convolution2D(filters=6, kernel_size=(5, 5), strides=(1, 1), activation='relu',input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3])))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Convolution2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	model.add(Flatten())
	model.add(Dense(units=120, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(units=84, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))

	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# define the larger model
def larger_model2():
	# create model
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(16, 5, 5, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', init='lecun_normal'))
	model.add(Dropout(0.2))
	model.add(Dense(84, activation='relu', init='lecun_normal'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# define the larger model
def larger_model_orig():
	# create model
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, border_mode='valid', input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(50, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

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

# build the model
model = larger_model2()
# Fit the model
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=30, batch_size=192, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

deep_utils.plot_accuracy(history)
deep_utils.plot_loss(history)
deep_utils.save_model(model,serialize_type='yaml',model_name='facial_recognition_large_cnn_model')
