#!/usr/bin/python3

# From https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/

# organize imports
#from __future__ import print_function
import numpy as np
#import h5py
import os
import io
import json
#import pickle
#import seaborn as sns
import matplotlib.pyplot as plt
import psutil
import time
import pickle

#from sklearn.metrics import classification_report
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing import image
from keras.models import Model
#from keras.models import model_from_json
#from keras.layers import Input
from keras.applications import mobilenet
from keras.utils.generic_utils import CustomObjectScope



#
# load the user configs
# 
config = json.load(io.open('tfk_conf.json', 'r', encoding='utf-8-sig'))

def ld_conf_path(f, config=config, opath="out_path"):
    p = os.path.join(config[opath], config[f])
    return p

# config variables
weights       = config["weights"]
include_top   = config["include_top"]
train_path    = config["train_path"]
test_path     = config["test_path"]
seed          = config["seed"]
batch_size    = config["batch_size"]
img_side_len  = config["img_side_len"]
out_path      = config["out_path"]
image_size = (img_side_len, img_side_len)
image_size_c = (img_side_len, img_side_len, 3)




start = time.time()
#
# Load the pre-trained model (without top layer)
#
p = os.path.join(out_path, 'xfer_train.h5')
with CustomObjectScope({'relu6': mobilenet.relu6,'DepthwiseConv2D': mobilenet.DepthwiseConv2D}):
    model = models.load_model(p)
print("Model loaded from " +p)
#model.summary()

end = time.time()
print('Elapsed time to load model: ' +str(end-start))

batch_size = 200
#
# Setup the Data generators
#
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
test_data = test_gen.flow_from_directory(
    test_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="sparse")
for image_batch,label_batch in test_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break
N_labels = test_data.labels.max() +1

start = time.time()
print('Classifying a batch of ' +str(batch_size) +' images...')
result_batch = model.predict(image_batch)
result_top = np.argmax(result_batch, axis=-1)
#labels_batch = imagenet_labels[result_top]
#print(result_top)
#print('Top 1 labels: ' +str(labels_batch))

# NOTE: label strings stored in test_data.class_indices

end = time.time()
print('Elapsed time to classify ' +str(batch_size) +' images: ' +str(end-start))

print('Scoring Top...')
correct = 0
for n in range(batch_size):
    if result_top[n] == label_batch[n]:
        correct += 1


accuracy =  correct / float(batch_size)
print("accuracy Top 1: " +str(accuracy))

#print('Failures:')
#for n in range(cats.shape[0]):
#    if cats[n] == 0.0:
#        print('Example # ' +str(n) +' classified as ' +str(imagenet_labels[result_top[n]]))


