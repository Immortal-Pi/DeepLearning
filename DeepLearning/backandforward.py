import numpy as np
import matplotlib.pyplot as plt


class Neural_Network():
    def __init__(self):
        self.inputsize=2
        self.hiddensize=3
        self.outputsize=1

        self.w1=np.random.randn(self.inputsize,self.hiddensize)
        self.w2=np.random.randn(self.hiddensize,self.outputsize)


    def sigmoid(self,s):
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self,s):
        #derivative function
        return s*(1-s)

    def forward(self,x):
        self.z=np.dot(x,self.w1)
        self.z2=self.sigmoid(self.z)
        self.z3=np.dot(self.z2,self.w2)
        o=self.sigmoid(self.z3)
        return o

    def backward(self,x,y,o):
        self.o_error=y-o
        self.o_delta=self.o_error*self.sigmoidPrime(o)

        self.z2_error=self.o_delta.dot(self.w2.T)
        self.z2_delta=self.z2_error*self.sigmoidPrime(self.z2)

        self.w1+=x.T.dot(self.z2_delta)
        self.w2+=self.z2.T.dot(self.o_delta)


    def train(self,x,y):
        o=self.forward(x)
        self.backward(x,y,o)

NN=Neural_Network()
loss=[]
epochs=400

x = np.array([[5, 40],
              [8, 82],
              [6, 52]], dtype=float)
# y: Target
y = np.array([[15], [24], [18]], dtype=float)

x=x/np.max(x,axis=0)
y=y/np.max(y)

for i in range(epochs):
    loss.append(np.mean(np.square(y-NN.forward(x))))
    NN.train(x,y)

print(loss)

plt.plot(np.arange(0,400),loss)
plt.show()
