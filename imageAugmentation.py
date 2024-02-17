import json
import os.path

import albumentations as alb
import cv2
import numpy as np

augmentor=alb.Compose(
    [alb.RandomCrop(width=450,height=450),
     alb.HorizontalFlip(p=0.5),
     alb.RandomBrightnessContrast(p=0.2),
     alb.RandomGamma(p=0.2),
     alb.RGBShift(p=0.2),
     alb.VerticalFlip(p=0.5)],bbox_params=alb.BboxParams(format='albumentations',label_fields=['class_labels'])
                    )

img=cv2.imread(os.path.join('data','train','images','3a2f4ffc-babc-11ee-a0ad-b42e996e5f84.jpg'))

with open(os.path.join('data','train','labels','3a2f4ffc-babc-11ee-a0ad-b42e996e5f84.json'),'r') as file:
    label=json.load(file)

print(label['shapes'][0]['points'])
coords=[0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

coords=list(np.divide(coords,[640,480,640,480]))
print(coords)

augmented=augmentor(image=img,bboxes=[coords],class_labels=['face'])
print(augmented['image'].shape)

