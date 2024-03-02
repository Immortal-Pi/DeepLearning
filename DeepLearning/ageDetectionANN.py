import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import utils
from keras.models import Sequential
from keras.layers import Dense,Flatten,InputLayer
import keras
import imageio
from PIL import Image


#get the data
train=pd.read_csv('datasets/train.csv')
test=pd.read_csv('datasets/test.csv')

print(train)
#display random movie character with age group