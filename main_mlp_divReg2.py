from __future__ import print_function

from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift, estimate_bandwidth
import scipy.cluster.hierarchy as hcluster
import scipy.cluster.hierarchy as hac
import scipy.cluster.hierarchy as fclusterdata

import time
from sklearn.preprocessing import normalize
import os

import numpy as np
import keras
from keras import backend as K
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop, SGD
from keras.wrappers.scikit_learn import KerasClassifier

def cluster_mask_agglo(W, threshold):
    ''' Clusters columns of weight matrix W and returns
        the number of distinct groups with average intra-cluster
        similarities above threshold.
    '''
    t0 = time.time()
    W = W.T
    threshold =  1.0-threshold   # Convert similarity to distance measure
    clusters = hcluster.fclusterdata(W, threshold, criterion="distance", metric='euclidean', depth=1, method='centroid')
    z = hac.linkage(W, metric='cosine', method='complete')
    labels = hac.fcluster(z, threshold, criterion="distance")
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    elapsed_time = time.time() - t0
    return n_clusters_

class RedundancyDropoutCallback(keras.callbacks.Callback):
    '''Adaptive dropout layer.
    Computes dropout rate and adapts its value based on how many filters are
    redundant at that particular layer and epoch.
    '''
    def __init__(self, threshold):
        self.threshold = threshold
    def on_epoch_end(self, epoch, logs={}):
        num = 0
        for i in range(len(self.model.layers)-1):
            if type(self.model.layers[i])==Dense:
                weight = self.model.layers[i].get_weights()[0]
                normed_weight = normalize(weight, axis=0, norm='l2')
                ## Save the pairwise similarities in epochs 1, 5, 30, 60, 100, 150, 200
                if epoch in {1, 5, 30, 60, 100, 150, 200}:
                    sim_matrix = np.dot(normed_weight.T,normed_weight)
                    kk=0
                    sim=[]
                    for ii in range(sim_matrix.shape[0]):
                        kk +=1
                        for jj in range(sim_matrix.shape[0]-kk):
                            jj = jj+kk
                            sim.append(sim_matrix[ii][jj])
                    sim_matrix_epoch.append(sim)
                ## Cluster filters
                num_redundant_filters = cluster_mask_agglo(weight, self.threshold)
                nonredun_filters.append(num_redundant_filters)
                print('\n nonredundant kernels in layer', i, 'is', num_redundant_filters)
                num +=num_redundant_filters
            if type(self.model.layers[i])==Dropout:
                self.model.layers[i].rate = K.variable(self.model.layers[i].rate , name='rate')
                print('\n previous dropout rate of layer ', i-1, 'is', K.eval(self.model.layers[i].rate))
                ## Compute dropout rate
                tau = 1.0-(1.0*num_redundant_filters/(weight.shape[1]))
                drop_out_ratio.append(tau)
                ## Set the dropout rate of current dropout layer
                K.set_value(self.model.layers[i].rate, tau)
                print('\n current dropout rate of layer ', i-1, 'is ', K.eval(self.model.layers[i].rate))
        print('\n total nonredundant kernels', num)

def DivRegularization(weight_matrix, p = 10, epsilon=0.3):
    ''' Diversity regularizer of neural network weights .
    '''
    weight_matrix=K.transpose(weight_matrix)
    weight_matrix_norm = K.l2_normalize(weight_matrix,axis=-1)
    sim_matrix = K.dot(weight_matrix_norm, K.transpose(weight_matrix_norm))
    sim_matrix = sim_matrix*(K.ones_like(sim_matrix)-K.eye(K.eval(sim_matrix).shape[0]))  # removing the diagonal element
    boolean_mask = K.cast(K.greater_equal(K.abs(sim_matrix), epsilon), 'float32')
    return   p *0.5* K.sum(K.square(K.abs(sim_matrix*boolean_mask)))

batch_size = 128
num_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test sa0.5mples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
hiddenSizeL1 = 1024
hiddenSizeL2 = 1024

model.add(Dense(hiddenSizeL1, activation='relu', input_shape=(784,),kernel_regularizer=DivRegularization,kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
model.add(Dropout(0.5))
model.add(Dense(hiddenSizeL2, activation='relu', kernel_regularizer=DivRegularization, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.summary()
epochs = 300

##### Hyperparameters for SGD
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

adaptdrop = RedundancyDropoutCallback(threshold=0.1)

callbacks_list = [adaptdrop]

global nonredun_filters
global drop_out_ratio
global sim_matrix_epoch
nonredun_filters = []
drop_out_ratio = []
sim_matrix_epoch = []

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test), callbacks=callbacks_list)

print(max(history.history['val_acc']))
