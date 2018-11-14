import warnings
warnings.filterwarnings("ignore")

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,SGD
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import h5py

num_classes = 10
epochs = 20
eta=1e-2
batch_size=200
alpha=1e-2
# the data, shuffled and split between train and test sets
data_path = 'data/'


# Read in the data
mnist_train=h5py.File(data_path+'mnist_train.h5','r')
X_train=mnist_train['X_train'][:].reshape((-1,28*28))
y_train=mnist_train['y_train']
mnist_test=h5py.File(data_path+'mnist_test.h5','r')
X_test=mnist_test['X_test'][:].reshape((-1,28*28))
y_test=mnist_test['y_test']

X_train = X_train[:]/255
X_test = X_test[:]/255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train[:], num_classes)
y_test_c = keras.utils.to_categorical(y_test[:], num_classes)

model = Sequential()
model.add(Dense(10, activation='softmax', input_shape=(784,)))

model.summary()

sgd = SGD(lr=eta, decay=0, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(X_test, y_test_c))
score = model.evaluate(X_test, y_test_c, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
y_pred=model.predict(X_test)
y_pred=y_pred.argmax(axis=1)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))