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
model_name    = config["model"]
weights       = config["weights"]
depth_mult    = config["depth_mult"]
width_mult    = config["width_mult"]
#include_top   = config["include_top"]
#train_path    = config["train_path"]
test_path     = config["test_path"]
seed          = config["seed"]
batch_size    = config["batch_size"]
num_batch     = config["num_batch"]
#img_side_len  = config["img_side_len"]
out_path      = config["out_path"]
#image_size = (img_side_len, img_side_len)
#image_size_c = (img_side_len, img_side_len, 3)
#batch_size = 5   # Debug
model_name = model_name.lower()

start = time.time()
#
# Load the pre-trained model (without top layer)
#
# TODO: Test depth_multiplier = 1.4, 0.35
# TODO: Test alpha (width multiplier) 
#model = tf.keras.applications.mobilenet.MobileNet(include_top=True, weights='imagenet',
#model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet',
#        input_tensor=tf.keras.layers.Input(shape=image_size_c),
#        input_shape=image_size_c)

defargs = {
        'include_top': True,
        'weights': weights }
if model_name == "vgg16":
    model = tf.keras.applications.VGG16(**defargs)
elif model_name == "vgg19":
    model = tf.keras.applications.VGG19(**defargs)
elif model_name == "resnet50":
    model = tf.keras.applications.ResNet50(**defargs)
elif model_name == "inceptionresnetv2":
    model = tf.keras.applications.InceptionResNetV2(**defargs)
elif model_name == "inceptionv3":
    model = tf.keras.applications.InceptionV3(**defargs)
elif model_name == "mobilenet":
    model = tf.keras.applications.MobileNet(**defargs,
        depth_multiplier=depth_mult,
        alpha=width_mult)
elif model_name == "mobilenetv2":
    model = tf.keras.applications.MobileNetV2(**defargs,
        depth_multiplier=depth_mult,
        alpha=width_mult)
elif model_name == "xception":
    model = tf.keras.applications.Xception(**defargs)
else:
    print ("Unknown model_name " +model_name +". No model loaded")
    exit()

image_size = (model.input_shape[1], model.input_shape[2])
 
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
#print("loaded {} files".format(len(test_data.filenames)))
#print("loaded {} ".format(test_data.filenames))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
end = time.time()
delta = end - start
print ("[INFO] model loaded and compiled in {0:.3f} seconds, used mem: {1}%".format(
    delta, psutil.virtual_memory().percent))

# Test the model
start = time.time()

results = np.array([])
labels = np.array([])
for nb in range(num_batch):
    for td_batch, td_batch_label in test_data:
        print("Image batch shape: ", td_batch.shape)
        print("Label batch shape: ", td_batch_label.shape)
        print("Batch number: " +str(nb+1) +" of " +str(num_batch))
        break

    results_batch = model.predict(
        td_batch,
        batch_size=batch_size,
        steps =1) # Note: Keeping steps=1 so that we can retrieve the labels of each batch

    if results.size == 0:
        results = results_batch
        labels  = td_batch_label
    else:
        results = np.concatenate( (results,results_batch) )
        labels  = np.concatenate( (labels,td_batch_label) )

end = time.time()
delta = end - start
print ("[INFO] evaluation completed in {0:.3f} seconds, used mem: {1}%".format(
    delta, psutil.virtual_memory().percent))
 
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
accuracy = lt.accuracy( results_top, labels)
print("Top 1 accuracy = {0:.3f}".format(accuracy))

accuracy = lt.accuracy( results_topk, labels)
print("Top {0} accuracy = {1:.3f}".format(K,accuracy))

accuracy = lt.accuracy( results_topl, labels)
print("Top {0} accuracy = {1:.3f}".format(L,accuracy))


end = time.time()
print('Elapsed time to classify {} images: {}'.format(
    num_batch*batch_size, end-start) )

