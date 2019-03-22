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
#import keras
#from keras import models
#from keras import layers
#from keras import optimizers
#from keras.preprocessing import image
#from keras.models import Model
#from keras.models import model_from_json
#from keras.layers import Input
#from keras.applications.mobilenet import MobileNet, preprocess_input
import pred_anal

#
# load the user configs
# 
config = json.load(io.open('tfk_conf.json', 'r', encoding='utf-8-sig'))

def ld_conf_path(f, config=config, opath="out_path"):
    p = os.path.join(config[opath], config[f])
    return p

# config variables
#weights       = config["weights"]
#include_top   = config["include_top"]
#train_path    = config["train_path"]
test_path     = config["test_path"]
seed          = config["seed"]
batch_size    = config["batch_size"]
img_side_len  = config["img_side_len"]
out_path      = config["out_path"]
image_size = (img_side_len, img_side_len)
image_size_c = (img_side_len, img_side_len, 3)
#batch_size = 5   # Debug

start = time.time()
#
# Load the pre-trained model (without top layer)
#
# TODO: Test depth_multiplier = 1.4, 0.35
# TODO: Test alpha (width multiplier) 
#model = tf.keras.applications.mobilenet.MobileNet(include_top=True, weights='imagenet',
model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet',
        input_tensor=tf.keras.layers.Input(shape=image_size_c),
        input_shape=image_size_c)

 
# Show a summary of the model. Check the number of trainable parameters
#model.summary()
print("[INFO] used mem: {}% - ".format(psutil.virtual_memory().percent) )

 
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True)
N_test_labels = test_data.labels.max() +1

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
end = time.time()
delta = end - start
print ("[INFO] model loaded and compiled in {0:.3f} seconds".format(delta))
print("[INFO] model compiled, used mem: {}% - ".format(psutil.virtual_memory().percent) )

# Test the model
start = time.time()

for td_batch, td_batch_label in test_data:
  print("Image batch shape: ", td_batch.shape)
  print("Label batch shape: ", td_batch_label.shape)
  break

results = model.predict(
        td_batch,
        batch_size=batch_size,
        steps =1)
end = time.time()
delta = end - start
print ("[INFO] evaluation completed in {0:.3f} seconds".format(delta))
print("[INFO] model trained, used mem: {}% - ".format(psutil.virtual_memory().percent) )
 
# Check Performance
#
results_top = np.argmax(results, axis=-1)
K = 5
L = 10
results_topk = pred_anal.result_top_k(results, K)
results_topl = pred_anal.result_top_k(results, L)
#labels_batch = imagenet_labels[result_top]
#print(result_top)
#print('Top 1 labels: ' +str(labels_batch))


print('Scoring Top...')
lt = pred_anal.label_types(test_data.class_indices )
accuracy = lt.accuracy( results_top, td_batch_label)
print("Top 1 accuracy = {0:.3f}".format(accuracy))

accuracy = lt.accuracy( results_topk, td_batch_label)
print("Top {0} accuracy = {1:.3f}".format(K,accuracy))

accuracy = lt.accuracy( results_topl, td_batch_label)
print("Top {0} accuracy = {1:.3f}".format(L,accuracy))


end = time.time()
print('Elapsed time to classify ' +str(batch_size) +' images: ' +str(end-start))

