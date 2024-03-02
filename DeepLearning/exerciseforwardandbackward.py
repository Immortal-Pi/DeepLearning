import numpy as np
import matplotlib.pyplot as plt


class Neural_Network(object):
    def __init__(self):
        self.inputsize=3
        self.hiddensize1=3
        self.hiddensize2=3
        self.outputsize=1

        self.w1=np.random.randn(self.inputsize,self.hiddensize1)
        self.w2=np.random.randn(self.hiddensize1,self.hiddensize2)
        self.w3=np.random.randn(self.hiddensize2,self.outputsize)


    def sigmoid(self,s):
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self,s):
        return s*(1-s)


    def forward(self,x):
        self.z=np.dot(x,self.w1)
        self.z2=self.sigmoid(self.z)
        self.z3=np.dot(self.z2,self.w2)
        self.z4=self.sigmoid(self.z3)
        self.z5=np.dot(self.z4,self.w3)
        o=self.sigmoid(self.z5)
        return o


    def backward(self,x,y,o):
        self.o_error=y-o
        self.o_delta= self.o_error*self.sigmoidPrime(o)

        self.z4_error=self.o_delta.dot(self.w3.T)
        self.z4_delta=self.z4_error*self.sigmoidPrime(self.z4)

        self.z2_error=self.z4_delta.dot(self.w2.T)
        self.z2_delta=self.z2_error*self.sigmoidPrime(self.z2)

        self.w1+=x.T.dot(self.z2_delta)
        self.w2+=self.z2.T.dot(self.z4_delta)
        self.w3+=self.z4.T.dot(self.o_delta)

    def train(self,x,y):
        o=self.forward(x)
        self.backward(x,y,o)



x=np.array(
    [
        [5,54,24],
        [8,68,6],
        [9,91,100]
    ],dtype=float
)

y=np.array(
    [ [12],[16],[42]

    ],dtype=float
)

#scaling
x=x/np.max(x,axis=0)
y=y/np.max(y)
NN=Neural_Network()
loss=[]
epochs=50

for i in range(epochs):
    loss.append(np.mean(np.square(y-NN.forward(x))))
    NN.train(x,y)
print(loss)

plt.plot(np.arange(0,50),loss)
plt.show()