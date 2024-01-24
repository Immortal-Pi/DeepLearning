import os
import time
import uuid #unique identifier
import cv2

IMAGE_PATH = os.path.join('data','images')
number_images=30

camera=cv2.VideoCapture(0)

for imgnum in range(number_images):
    print('Collecting images {}'.format(imgnum))
    success, frame=camera.read()
    imgname = os.path.join(IMAGE_PATH,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname,frame)
    cv2.imshow('frame',frame)
    time.sleep(0.5)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
