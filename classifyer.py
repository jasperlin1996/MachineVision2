import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
import os
import preprocess
import matplotlib.pyplot as plt
np.random.seed(0)
####################
(X_train, y_train), (X_test, y_test) = preprocess.load_data(0.1)
print("type of training data : {}".format(type(X_train)))
print("shape of training data : {}".format(X_train.shape))
print(y_test.shape)
#X_train is 60000 rows of 28x28 values and is reshaped in 60000 x 784
d           = X_train.shape[1]*X_train.shape[2]

#
#X_train = X_train.reshape(X_train.shape[0], d)
#print("X_test.shape : {}".format(X_test.shape))
#X_test  = X_test.reshape(X_test.shape[0], d)
#print("X_test.shape : {}".format(X_test.shape))


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

####################
#-#
####################
# normalize 
X_train /= 255
X_test  /= 255


output_unit = 12   # number of outputs = number of digits
hidden_unit = 128

# one-hot-encoding
Y_train = np_utils.to_categorical(y_train, output_unit)
Y_test = np_utils.to_categorical(y_test, output_unit)
print("X_train.shape : {}".format(X_train.shape))
print("X_test.shape : {}".format(X_test.shape))
print("Y_train.shape : {}".format(Y_train.shape))
print("Y_test.shape : {}".format(Y_test.shape))

####################
model = Sequential()

model.add(Convolution2D(
    batch_input_shape=(64,1,34,34),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first',
))
model.add(Activation('relu'))

model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first',
))

model.add(Convolution2D(64, 5, strides=1, padding='same', data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(2, 2, 'same', data_format='channels_first'))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

model.add(Dense(12))
model.add(Activation('softmax'))

adam = Adam(lr = 1e-4)

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
print("Training -------------")
#training
model.fit(X_train, Y_train, epochs=10, batch_size=64)
print("\nTesting --------------")
loss ,accuracy = model.evaluate(X_test, Y_test)
print("\nTest loss: {}, \t Test accuracy: {}".format(loss, accuracy))