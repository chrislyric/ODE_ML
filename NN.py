
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import h5py
import keras
import sys
import os
import pandas as pd
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from keras.models import load_model
from keras.regularizers import Regularizer

# Parameters
LEARNING_RATE = 0.01
BATCH_SIZE = 50
EPOCHS = 100
scale = 1

input_size = 1
output_size = 1
nodes = 3
input_file = "input_exp1.csv"
output_file = "output_exp1.csv"

model_name = "./exp1_node_3"
model_file = model_name+'/model.hdf5'
weight_file = model_name+'/best_weights.h5'



def CreateModel(input_shape,output_shape):
    input_para = Input((input_shape,), name = 'input_para')
    x = Dense(nodes,activation = 'tanh')(input_para)
    x= Dense(nodes,activation = 'tanh')(x)
    output = Dense(output_shape,activation = 'tanh')(x)
    
    model = Model(inputs = input_para, outputs = output, name = 'Mode')
    
    return model


def read_data_from_file(feature_num,label_num,filename_x, filename_y,testing_ratio = 0.02):
    FEATURES = []
    LABELS = []

    for i in range (feature_num):
        FEATURES.append(str(i))
    for j in range (label_num):
        LABELS.append('a'+str(j))
    raw_feature = pd.read_csv(filename_x, header=None,skipinitialspace=False,names=FEATURES)
    raw_label = pd.read_csv(filename_y, header=None,skipinitialspace=False,names=LABELS)

# Split all samples into to category: for_training and for_testing
    testing_number = int(testing_ratio*raw_feature.shape[0])

    print('there are {} testing samples, and {} training samples'.format(testing_number,raw_feature.shape[0]-testing_number))
    import random
# testing_samples_index = [37,235,908,72,767,905,715,645,847,960]
    testing_samples_index = []
    training_samples_index = []
    random.seed(a=1)
    for i in range(testing_number):
        test_ind = random.randint(0,raw_feature.shape[0])
        testing_samples_index.append(test_ind)
        print(test_ind)
    training_samples_index = list(set(range(raw_feature.shape[0])).difference(set(testing_samples_index)))

    test_x = pd.DataFrame(raw_feature.loc[testing_samples_index, :].mul(scale), columns =FEATURES,dtype = np.float32).as_matrix()
    test_y =  pd.DataFrame(raw_label.loc[testing_samples_index, :].mul(scale)).as_matrix()
    train_x = pd.DataFrame(raw_feature.loc[training_samples_index, :].mul(scale),columns =FEATURES,dtype = np.float32).as_matrix()
    train_y = pd.DataFrame(raw_label.loc[training_samples_index,:].mul(scale)).as_matrix()

    return train_x, train_y, test_x, test_y, testing_samples_index



def print_predict_to_file(pred_y,testing_samples_index):
    for i in range(len(testing_samples_index)):
        pred_i = pred_y[i,:]
        filename = './{}/{}_prediction.csv'.format(model_name,str(testing_samples_index[i]))
        np.savetxt(filename,pred_i,delimiter = ",")
        


def main(argv):
    # Data processing
    directory = './'+model_name
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    train_x, train_y, test_x, test_y, testing_samples_index = read_data_from_file(input_size,output_size,input_file, output_file)
    test_sample_name = './'+model_name+'/sample_index.txt'
    np.savetxt(test_sample_name,testing_samples_index,fmt = '%i',delimiter = '\n')
    # Load or create model
    try:
        model = load_model(model_file)
        model.load_weights(weight_file)
    except:
        model = CreateModel(input_size,output_size)
        sgd = SGD(lr=LEARNING_RATE)
        model.compile(loss = 'mse', optimizer = sgd, metrics = ['mse'])
    
    model.summary()
    checkpointer = ModelCheckpoint(filepath = weight_file,
                                   monitor = 'val_loss',
                                   verbose = 0,
                                   save_best_only = True,
                                   save_weights_only = False,
                                   mode = 'auto')
    history = model.fit(train_x, train_y,
                        epochs = EPOCHS,
                        batch_size = BATCH_SIZE,
                        shuffle = True,
                        verbose = 1,
                        callbacks = [checkpointer],
                        validation_data = (test_x, test_y))
    model.save(model_file)
    
    # Plotting training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train','Test'], loc = 'upper right')
    plt.show()
    
    pred_y = model.predict(test_x)
    print_predict_to_file(pred_y, testing_samples_index)


if __name__=="__main__":
    main(sys.argv)

