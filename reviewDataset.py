import tensorflow as tf
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt

# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu,True)
# tf.config.list_physical_devices('GPU')

images=tf.data.Dataset.list_files('data\\images\\*.jpg',shuffle=False)
# print(images.as_numpy_iterator().next())

def load_image(x):
    byte_img=tf.io.read_file(x)
    img=tf.io.decode_image(byte_img)
    return img


images = images.map(load_image)
# print(images.as_numpy_iterator().next())
image_generator=images.batch(4).as_numpy_iterator()
plot_images=image_generator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()


