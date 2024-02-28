import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

class neural_network(object):
    def __init__(self):
        self.inputSize=2
        self.outputSize=1
        self.hiddenSize=3


#forward propagation and back propagation

# X: (Feature 1, Feature 2)
x = np.array([[5, 40],
              [8, 82],
              [6, 52]], dtype=float)
# y: Target
y = np.array([[15], [24], [18]], dtype=float)

#scaling units
x=x/np.max(x,axis=0)
y=y/np.max(y)

#take some known weights
fuel = 5/8
dist = 40/82
budget = 15/24

#consider some weights
fuelw1, fuelw2, fuelw3= 0.3,0.2,0.6
distw1, distw2, distw3= 0.22, 0.56, 0.7


