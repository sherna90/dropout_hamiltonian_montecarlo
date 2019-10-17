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
epochs = 100
eta=1e-2
batch_size=256
alpha=1e-3
# the data, shuffled and split between train and test sets
data_path = 'data/'


# Read in the data
plants_train=h5py.File(data_path+'train_features_labels.h5','r')
X_train=plants_train['train_features']
y_train=plants_train['train_labels']
plants_test=h5py.File(data_path+'validation_features_labels.h5','r')
X_test=plants_test['validation_features']
y_test=plants_test['validation_labels']

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train[:], num_classes)
#y_test_c = keras.utils.to_categorical(y_test[:], num_classes)

model = Sequential()
model.add(Dense(38, activation='softmax', input_shape=(X_train.shape[1],)))

model.summary()

sgd = RMSprop(lr=eta, decay=1e-6, rho=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    shuffle="batch",
                    epochs=epochs,
                    verbose=False)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
y_pred=model.predict(X_test)
y_pred=y_pred.argmax(axis=1)
y_test_c=np.argmax(y_test[:],axis=1)
print(classification_report(y_test_c, y_pred))
print(confusion_matrix(y_test_c, y_pred))

plants_train.close()
plants_test.close()