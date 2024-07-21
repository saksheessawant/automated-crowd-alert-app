import tensorflow as tf
import keras
import os
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.models import load_model
classifier = tf.keras.models.load_model(r'C:\Users\91992\Downloads\Bus.h5')

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


import numpy as np
from google.colab import files
from keras.preprocessing import image


test_image = image.load_img('/content/drive/My Drive/TestImages/pic (4807).jpg', target_size = (64,64))
x = image.img_to_array(test_image)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = classifier.predict(images, batch_size=32)
  
  ##cv2.imshow(fn)
  #print(max(classes))
Labels=['High','Low','Moderate','Very High','Very Low']   ### please write your classes names
index=np.argmax(classes)
print(Labels[index])