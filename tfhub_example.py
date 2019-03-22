#!/usr/bin/python3

# Example based on https://www.tensorflow.org/tutorials/images/hub_with_keras

# Data from:
# https://www.kaggle.com/c/dogs-vs-cats
# https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data


import tensorflow as tf
#import matplotlib.pylab as plt
import numpy as np
import PIL.Image as Image
import time

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
#TODO figure out why 'from keras import layers' does not work (layers.Lambda)
#from keras import models
#from keras import optimizers

start = time.time()

print(tf.version)
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

print('Classifier config...')
# Setup the classifier
# See a list of nets at: https://tfhub.dev/s?network-architecture=mobilenet-v2
#classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/classification/2" #@param {type:"string"}
#classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2" #@param {type:"string"}
classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_035_224/classification/2" #@param {type:"string"}
def classifier(x):
  classifier_module = hub.Module(classifier_url)
  return classifier_module(x)
# Use hub.module to load a mobilenet, and tf.keras.layers.Lambda to wrap it up as a keras layer. 
IMAGE_SIZE = hub.get_expected_image_size(hub.Module(classifier_url))
classifier_layer = layers.Lambda(classifier, input_shape = IMAGE_SIZE+[3])
model = tf.keras.Sequential([classifier_layer])
model.summary()

print('Initialize model...')
# When using Keras, TFHub modules need to be manually initialized.
import tensorflow.keras.backend as K
sess = K.get_session()
init = tf.global_variables_initializer()

sess.run(init)

#
#           Run it on a single image
#
"""
print('Testing...')
cat1 = Image.open('cat_photos/russian_blue/cat1.jpg').resize(IMAGE_SIZE)
cat1 = np.array(cat1)/255.0
#cat1.shape
result = model.predict(cat1[np.newaxis, ...])
print('result: ' +str(result))
#result.shape

# The result is a 1001 element vector of logits, rating the probability of each class for the image.
# So the top class ID can be found with argmax:
predicted_class = np.argmax(result[0], axis=-1)
print("Predicted imagenet class: " +str(predicted_class))
"""
# We have the predicted class ID, Fetch the ImageNet labels, and decode the predictions
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())
#predicted_class_name = imagenet_labels[predicted_class]
#print("Prediction: " + predicted_class_name)

end = time.time()
print('Elapsed time to load classifier : ' +str(end-start))

#
#           Run it on a batch of images
#
exec(open("./tfhub_test.py").read())

