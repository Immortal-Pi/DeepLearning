#get the data
#view sample images and classes
#transforming the dataset to a 1d array after reshaping all the images 32*32*3
#normalize the data
#knowing the distribution of class in data
#encoding the categorical variable to numeric
#building deep neural network
#predicting
#visualize inspection of prediction

import os

import numpy as np
from keras import Sequential
import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense,Flatten,Softmax,InputLayer


train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')

img_path=os.path.join('datasets/Train',train['ID'])
img=np.array()