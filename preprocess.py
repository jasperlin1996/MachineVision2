from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
import random
path = "D:\\HW2\\data_set"

dirs = listdir(path)

X_train = []
Y_train = []

def preprocess():
    for d in dirs:
        dirpath = join(path, d)
        print(d)
        files = listdir(dirpath)
        for f in files:
            filepath = join(dirpath, f)
            X_train.append(cv2.imread(filepath))
            Y_train.append(int(d)-1)
def load_data(percentage):
    X_test = []
    Y_test = []
    preprocess()
    X_ret = np.array(X_train)[:,:,:,0]
    Y_ret = np.array(Y_train)
    #ret = np.reshape(ret,(2902,34,34,3))
    print(X_ret.shape)
    print(Y_ret.shape)
    print(Y_ret[400:410])
    for i in reversed(range(0, int(X_ret.shape[0]*(1)))):
        if random.random() < percentage:
            X_test.append(X_ret[i])
            Y_test.append(Y_ret[i])
            X_ret = np.delete(X_ret,i,axis = 0)
            Y_ret = np.delete(Y_ret,i,axis = 0)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    print(X_ret.shape)
    print(Y_ret.shape)
    print(X_test.shape)
    print(Y_test.shape)
    return (X_ret, Y_ret), (X_test, Y_test)

if __name__ == '__main__':
    load_data(0.1)