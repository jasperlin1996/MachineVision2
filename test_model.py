import os
       
# Make Keras use the TensorFlow backend. This statement must be executed before importing Keras.
if 'KERAS_BACKEND' in os.environ and os.environ['KERAS_BACKEND'] != 'tensorflow':
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    
#from __future__ import print_function
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
import preprocess
import h5py
np.random.seed(0)   


(X_train, y_train), (X_test, y_test) = preprocess.load_data(0.1)
print("type of training data : {}".format(type(X_train)))
print("shape of training data : {}".format(X_train.shape))
print(y_test.shape)
#X_train is 60000 rows of 28x28 values and is reshaped in 60000 x 784
d           = X_train.shape[1]*X_train.shape[2]

#
X_train = X_train.reshape(X_train.shape[0], d)
print("X_test.shape : {}".format(X_test.shape))
X_test  = X_test.reshape(X_test.shape[0], d)
print("X_test.shape : {}".format(X_test.shape))


X_train = X_train.astype(np.float32)
X_test  = X_test.astype(np.float32)
y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)
print("y_train.shape : {}".format(y_train.shape))
print("y_test.shape : {}".format(y_test.shape))
# show some training samples
vis_train = X_train[np.random.permutation(X_train.shape[0])[:100],:].reshape((10,10,34,34)).transpose([0,2,1,3]).reshape((340,340))
vis_test = X_test[np.random.permutation(X_test.shape[0])[:100],:].reshape((10,10,34,34)).transpose([0,2,1,3]).reshape((340,340))

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.imshow(vis_train,cmap='gray')
plt.title('100 randomly selected training samples')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(vis_test,cmap='gray')
plt.title('100 randomly selected testing samples')
plt.axis('off')

plt.show()

print(X_test.shape)

##############################################################

# normalize 
X_train /= 255
X_test  /= 255


output_unit = 12   # number of outputs = number of digits
hidden_unit = 128

# one-hot-encoding
Y_train = np_utils.to_categorical(y_train, output_unit)
Y_test = np_utils.to_categorical(y_test, output_unit)

print("############")
for i in Y_test:
    print(i)

print("############")

print("X_train.shape : {}".format(X_train.shape))
print("X_test.shape : {}".format(X_test.shape))
print("Y_train.shape : {}".format(Y_train.shape))
print("Y_test.shape : {}".format(Y_test.shape))

# define network structure

model = Sequential()
model.add(Dense(hidden_unit, input_shape=(d,))) # the first hidden layer
model.add(Activation('relu')) # activation function is ReLU
model.add(Dense(hidden_unit)) # the second hidden layer
model.add(Activation('relu')) # activation function is ReLU
model.add(Dense(128)) # the second hidden layer
model.add(Activation('sigmoid')) # activation function is ReLU
model.add(Dropout(0.1))
model.add(Dense(128)) # the second hidden layer
model.add(Activation('tanh')) # activation function is ReLU
model.add(Dropout(0.1))
model.add(Dense(output_unit)) # 10 outputs
model.add(Activation('softmax')) # final stage is softmax
model.summary()

#train

numberEpoch= 100
batchSize = 64
valiation_ratio=0.1 # the ratio of training samples reserved for validation

optimizer  = SGD() # optimizer
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=batchSize, epochs=numberEpoch, verbose=True, validation_split=valiation_ratio)
print(X_test.shape)
print(Y_test.shape)
#after training, this model is evaluated by the test set
score = model.evaluate(X_test, Y_test, verbose=True)
print(X_test.shape)
#model.save('test_model.h5')
print("\nTest score:", score[0])
print('Test accuracy:', score[1])