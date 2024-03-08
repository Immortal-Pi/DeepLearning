#get data
#show random picture
#split the data test and train
#take the image data and normalize the data
#label encode
#define CNN
import numpy as np
#prediction and actual accuracy
#Confusion matrix visulization


import pandas as pd
import os
from PIL import Image
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split



#get data
label_data=pd.read_csv('datasets/cifar10Labels.csv',index_col=0)
print(label_data)



#show random picture
# idx=2424
# img_path=os.path.join('datasets/cifar10',f'{idx}.png')
# img=imageio.imread(img_path)
#
# plt.imshow(img)
# plt.show()


#split the data for
ytrain,ytest=train_test_split(label_data['label'],test_size=0.2,random_state=42)
# print(ytrain,ytest)

temp=[]
for i in ytrain.index:
    img_path=os.path.join('datasets/cifar10',f'{i}.png')
    img=np.array(imageio.imread(img_path)).astype('float32')
    temp.append(img)

xtrain=np.stack(temp)
print(xtrain)

for i in ytest.index:
    img_path=os.path.join('datasets/cifar10',f'{i}.png')


