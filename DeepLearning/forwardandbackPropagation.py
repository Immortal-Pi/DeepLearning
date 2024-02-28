import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

class Neural_Network(object):
    def __init__(self):
        self.inputSize=2
        self.outputSize=1
        self.hiddenSize=3


        #weights
        #weight matrix from input to hidden layer
        self.W1=np.random.randn(self.inputSize,self.hiddenSize)
        #weight matrix from hidden to output layer
        self.W2=np.random.randn(self.hiddenSize,self.outputSize)


    def sigmoid(self,s):
        return 1/(1+np.exp(-s))

    def forward(self,x):
        #forward propagation
        self.z = np.dot(x,self.W1) #dot product of x (input) and first set of 3*2 weights
        self.z2 = self.sigmoid(self.z) #activation function
        self.z3 = np.dot(self.z2,self.W2) #dot products of hidden layer (z2) and second set of 3*1 weights
        o = self.sigmoid(self.z3) #final activation function
        return o


    def sigmoidPrime(self,s):
        #derivative of sigmoid
        return s*(1-s)


    def backward(self,x,y,o):
        #backward popagate through the network
        self.o_error = y-o #error in output
        self.o_delta = self.o_error*self.sigmoidPrime(o) #apply derivate of sigmoid to error

        #z2 error: how much our hidden layer weights contribute to output error
        self.z2_error = self.o_delta.dot(self.W2.T)
        #applyying derivative of sigmoid to z2 error
        self.z2_delta = self.z2_error*self.sigmoidPrime(self.z2)


        #adjust first set (input --> hidden) weights
        self.W1+=x.T.dot(self.z2_delta)
        #adjusting second set(hidden --> output) weights
        self.W2+= self.z2.T.dot(self.o_delta)

    def train(self,x,y):
        o=self.forward(x)
        self.backward(x,y,o)


# X: (Feature 1, Feature 2)
x = np.array([[5, 40],
              [8, 82],
              [6, 52]], dtype=float)
# y: Target
y = np.array([[15], [24], [18]], dtype=float)


#scaling units
x=x/np.max(x,axis=0)
y=y/np.max(y)



NN=Neural_Network()

loss=[]
epochs=400

for i in range(epochs):
    loss.append(np.mean(np.square(y-NN.forward(x))))

    NN.train(x,y)

print(loss)


#visualilze
plt.plot(np.arange(0,400),loss)
plt.xlabel('epochs')
plt.ylabel('loss')

plt.show()

# #forward propagation and back propagation
#
# # X: (Feature 1, Feature 2)
# x = np.array([[5, 40],
#               [8, 82],
#               [6, 52]], dtype=float)
# # y: Target
# y = np.array([[15], [24], [18]], dtype=float)
#
# #scaling units
# x=x/np.max(x,axis=0)
# y=y/np.max(y)
#
# #take some known weights
# fuel = 5/8
# dist = 40/82
# budget = 15/24
#
# #consider some weights
# fuelw1, fuelw2, fuelw3= 0.3,0.2,0.6
# distw1, distw2, distw3= 0.22, 0.56, 0.7
#
#
# #with the above lets fin the values for hidden layers nodes
#
# HN1=(fuel*fuelw1)+(dist*distw1)
# HN2=(fuel*fuelw2)+(dist*distw2)
# HN3=(fuel*fuelw3)+(dist*distw3)
#
#
# #apply non linearity of these values to get the final hidden layer node values
# def sigmoid(s):
#     return 1/(1+np.exp(-s))
#
# sigH1=sigmoid(HN1)
# sigH2=sigmoid(HN2)
# sigH3=sigmoid(HN3)
#
# #consider some weight between hidden layer and output layer
# hw1,hw2,hw3=0.21,0.45,0.85
#
# #final value of output
# output=sigH1*hw1+sigH2*hw2+sigH3*hw3
# print(output)
#
#
#
# #back propagation to avoid loss
# #gradient descent help reduce loss
#
# def sigmoidPrime(s):
#     #derivative of sigmoid
#     return s* (1-s)
#
# def backward(self,x,y,o):
#     #backward propafation through the network
#     self.o_error=y-o
#     self.o_delta = self.o_error*self.sigmoidPrime(o)
#
#     self.z2_error=self.o_error*self.o_delta.dot(self.w2.T)
#     self.z2_delta=self.z2_error*self.sigmoidPrime(self.z2)
#
#     self.w1+=x.T.dot(self.z2_delta)
#     self.w2+=self.z2.T.dot(self.o_delta)
#
# backward(x,y,)