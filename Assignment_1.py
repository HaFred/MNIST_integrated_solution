#!/usr/bin/env python
# coding: utf-8

# - Please note that this ipynb is divided into 4 parts, each part contains the solution codes corresponding to a single problem in the assignment.
# when assigning varibales, follows the routine when there are similar names:
# dis_class means distances of classes, an array
# '_' denotes 'of'
# classDis means the dict of the distance of the class, a dict

import numpy as np
import pandas as pd
import operator
import timeit
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.data import loadlocal_mnist
from numba import jit, cuda 
import tensorflow as tf


# The naming of data variable following downhyphen: test_set test_img
train_img, train_label = loadlocal_mnist(
                        images_path='C:\\Users\\Hauru\\Dropbox\\19 Courses\\DL\\Assignment1\\train-images.idx3-ubyte',
                        labels_path='C:\\Users\\Hauru\\Dropbox\\19 Courses\\DL\\Assignment1\\train-labels.idx1-ubyte')
# train_img = np.int16(train_img) # int16 is for KNN
test_img, test_label = loadlocal_mnist(
                        images_path='C:\\Users\\Hauru\\Dropbox\\19 Courses\\DL\\Assignment1\\t10k-images.idx3-ubyte',
                        labels_path='C:\\Users\\Hauru\\Dropbox\\19 Courses\\DL\\Assignment1\\t10k-labels.idx1-ubyte')
# test_img = np.int16(test_img)
print(test_img.shape)

# - Proves that the imgs matches the labels well
print (train_img.shape[0],train_img.shape[1])
print (train_img[0])


# - Problem 1 - KNN

def SAD(X, Y, dimen): # Sum of Absolute Difference
    result = np.absolute(X - Y) # note that it cannot be uint8, otherwise complementary numbers
    distance_sum = result.sum(axis=dimen) # the less the distance is, the more similar two samples are
    return distance_sum    

# Proves that we have absolute diff, dimen=0 sum the columns, dim=1 sum the rows
tt = SAD(train_img, np.tile(train_img[0],(60000,1)), 1)
print (tt)

gg = SAD(train_img[1], train_img[0], 0)
print (gg)

class my_KNN_classifier2:  # using OOP constructors: __init__
    
    def __init__(self, inputX, dataSet, data_label, k):
        dataSetSize = dataSet.shape[0]
        self.distances = SAD(np.tile(inputX, (dataSetSize, 1)), dataSet ,dimen=1)
        self.sortedDistInd = self.distances.argsort()
        self.data_label = data_label
        self.k = k
            
    def K_count(self): # sortedClass gives back the detailed output of KNN 
        self.classCount = {}
        for i in range(self.k):
            voteLabel = self.data_label[self.sortedDistInd[i]]
            self.classCount[voteLabel] = self.classCount.get(voteLabel, 0)+1 # dict.get return the vL corresponding integer
        self.sortedClass = sorted(self.classCount.items(), reverse=True) #reverse means descending

def plot_my_KNN_test(test_set, k): 
    tic = timeit.default_timer()
    k_axis = range(1, 11)
    accu = np.zeros(k)
    for k_var in range(k):        
        opt = 0
        for i in range(test_set.shape[0]):
            answer1 = my_KNN_classifier2(test_set[i], train_img, train_label, k)
          #  neigh = answer1.sortedClass[0]
            answer1.K_count();
            pred = answer1.sortedClass[0][0] # pred w/o smallest aggregated SAD
            if pred == test_label[i]:
                opt+=1
        accu[k_var] = opt/test_set.shape[0]
    plt.plot(k_axis, accu, 's-', color = 'r', label = "Accu v.s. Num_K")
    plt.xlabel('Number of K')
    plt.ylabel('Accuracy')
    for a, b in zip(k_axis, accu):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=11)
    plt.legend(loc = 'best')
    plt.locator_params('y', nbins = 14)
    fig = plt.gcf()
    plt.show()
    fig.savefig('my_knn_accu_noSmall.jpg', dpi=200)
    toc = timeit.default_timer()
    print ('Elapsed: %s seconds'%(toc - tic))
    print (accu)
    return accu

plot_my_KNN_test(test_img, 10)

# - For my_KNN, when K=1
opt = 0
tic = timeit.default_timer()

test_img = test_img[0:10, :]

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_img, train_label)

for i in range(test_img.shape[0]):
    if neigh.predict(np.reshape(test_img[i],(1,784))) == test_label[i]:
        opt+=1
        
accu = opt/test_img.shape[0]
toc = timeit.default_timer()
print ('Elapsed: %s'%(toc - tic))
print (accu, opt)

def plot_sklearn_KNN_test(test_img, k): #scikit learn KNN test, plot k=1:k
    tic = timeit.default_timer()
    k_axis = range(1, 11)
    accu = np.zeros(k)
    for k_var in range(k):
        neigh = KNeighborsClassifier(n_neighbors = k_var + 1)
        neigh.fit(train_img, train_label)
        opt = 0        
        for i in range(test_img.shape[0]):
            if neigh.predict(np.reshape(test_img[i],(1,784))) == test_label[i]:
                opt+=1
        accu[k_var] = opt/test_img.shape[0]
        print (accu)
    plt.plot(k_axis, out, 's-', color = 'r', label = "Accu v.s. Num_K")
    plt.xlabel('Number of K')
    plt.ylabel('Accuracy')
    for a, b in zip(k_axis, out):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=11)
    plt.legend(loc = 'best')
    plt.locator_params('y', nbins = 14)
    fig = plt.gcf()
    plt.show()
    fig.savefig('scikit_knn_accu.jpg', dpi=200)
    toc = timeit.default_timer()  
    print ('Elapsed: %s seconds for scikit learn KNN'%(toc - tic))
    
plot_sklearn_KNN_test(test_img[0:1000,:], 10)


# - Problem 2 - MLP

import tensorflow as tf
train_img_NN = np.reshape(train_img, (60000,28,28)) / 255      # normalization, make it floating point
test_img_NN = np.reshape(test_img, (10000,28,28)) / 255
test_img_NN.dtype
from tensorflow.keras.layers import Conv2D, Dense, AveragePooling2D, Flatten, LeakyReLU

def MLP (num_neurons, test_set):

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(num_neurons,
                             activation = tf.nn.relu),
       tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_neurons,
                             activation = tf.nn.relu),
       tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation = tf.nn.softmax)
    ])

    model.compile(
        optimizer = 'adam',  # do not assign the learning rate as 0.01, dynamic lr would be better
        #tf.keras.optimizers.Adam,
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy'])

    model.fit(train_img_NN, train_label, 
              batch_size = 64, 
              epochs = 20, 
              #shuffle = 0,
#              validation_data = (train_img_NN[50000:60000,:,:], train_label[50000:60000]),
            validation_split = 1/6
             )

    num_eva = 10
    accu = np.zeros(num_eva)
    for i in range(num_eva):
        accu[i] = model.evaluate(test_set, test_label)[1]
    accuMean = np.mean(accu)
    print ('The accu is %s' %accuMean)
    
    return accuMean
    
def MLP_plot():
    tic = timeit.default_timer()
    num_neurons_list = np.array([4, 8, 16, 32, 64, 128, 256])
    accu = np.zeros(num_neurons_list.shape[0])
    for i in range(num_neurons_list.shape[0]):
        accu[i] = MLP(num_neurons_list[i], test_img_NN)
    accu = np.around(accu, decimals=4)
    plt.plot(num_neurons_list, accu, 's-', color = 'r', label = "Accu v.s. Num_neurons")
    plt.xscale('log', basex=2)
    plt.xlabel('Number of Neurons')
    plt.ylabel('Accuracy')
    for a, b in zip(num_neurons_list, accu):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=11)
    plt.locator_params('y', nbins = 14)
    fig = plt.gcf()
    plt.show()
    fig.savefig('MLP_accu.jpg', dpi=200)
    toc = timeit.default_timer()
    print ('Elapsed: %s seconds'%(toc - tic))
    return accu

accu = MLP_plot()

# - Problem 3 - LeNet

lenet_model = tf.keras.Sequential([
    Conv2D(6, 
           kernel_size=(5, 5), 
           strides=(1, 1), 
           activation='tanh', 
           input_shape=(28,28,1), 
           padding='same'),
    AveragePooling2D(pool_size=(2,2),
                     strides=(1,1),
                     padding='valid'),
    Conv2D(16, 
           kernel_size=(5, 5), 
           strides=(1, 1), 
           activation='tanh', 
           padding='valid'),
    AveragePooling2D(pool_size=(2,2),
                     strides=(1,1),
                     padding='valid'),
    Conv2D(120, 
           kernel_size=(5, 5), 
           strides=(1, 1), 
           activation='tanh', 
           padding='valid'),
    Flatten(),
    Dense(84,
         activation='tanh'),
    Dense(10,
         activation='softmax')    
])

lenet_model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
lenet_model.fit(x=np.reshape(train_img_NN, (60000, 28, 28, 1)), y=train_label, epochs=20, batch_size=64, validation_split=1/6)
accu_le = lenet_model.evaluate(np.reshape(test_img_NN, (10000, 28, 28, 1)),
                                             test_label)[1]
print(accu_le)
print('lenet_model sample predictions: %s \n' %lenet_model.predict_classes(np.reshape(test_img_NN[59:77,:,:],(18,28,28,1))))
print('the actual labels: %s' %test_label[59:77])
np.reshape((train_img_NN[59:77,:,:],(18,28,28,1)))
lenet_model.summary()
lenet_model.layers[7].output_shape


# - Problem 4 - CAN

can_model = tf.keras.Sequential([
    Conv2D(32, 
           kernel_size=(3, 3), 
#            activation='lrelu', 
           input_shape=(28,28,1), 
           padding='same',
            dilation_rate=(1,1)),
    LeakyReLU(alpha=0.1),
    Conv2D(32, 
           kernel_size=(3, 3), 
#            activation='lrelu', 
           padding='same',
            dilation_rate=(2,2)),
    LeakyReLU(alpha=0.1),
    Conv2D(32, 
           kernel_size=(3, 3), 
#            activation='lrelu', 
           input_shape=(28,28,1), 
           padding='same',
            dilation_rate=(4,4)),
    LeakyReLU(alpha=0.1),
    Conv2D(32, 
           kernel_size=(3, 3), 
#            activation='lrelu', 
           input_shape=(28,28,1), 
           padding='same',
            dilation_rate=(8,8)),
    LeakyReLU(alpha=0.1),
    Conv2D(10, 
           kernel_size=(3, 3), 
#            activation='lrelu', 
           input_shape=(28,28,1), 
           padding='same',
            dilation_rate=(1,1)),
    LeakyReLU(alpha=0.1),
    AveragePooling2D(pool_size=(2,2),
                     strides=(1,1),
                     padding='valid'),
    Flatten(),
#     Dense(84,
#          activation='tanh'),
    Dense(10,
         activation='softmax')    
])
can_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
can_model.fit(x=np.reshape(train_img_NN, (60000, 28, 28, 1)), y=train_label, epochs=20, batch_size=64, validation_split=1/6)
accu_can = []
accu_can = can_model.evaluate(np.reshape(test_img_NN, (10000, 28, 28, 1)),
                                             test_label)[1]
print(accu_can)
print('lenet_model sample predictions: %s \n' %can_model.predict_classes(np.reshape(test_img_NN[59:77,:,:],(18,28,28,1))))
print('the actual labels: %s' %test_label[59:77])

