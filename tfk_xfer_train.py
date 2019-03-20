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
from keras.layers import Input
from keras.applications.mobilenet import MobileNet, preprocess_input



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

#
# Load the pre-trained model (without top layer)
#
base_model = MobileNet(include_top=False, weights='imagenet',
        input_tensor=Input(shape=image_size_c),
        input_shape=image_size_c)

#
# Freeze the layers except the last 4 layers
#
#for layer in base_model.layers[:-4]:
for layer in base_model.layers:
    layer.trainable = False
# Check the trainable status of the individual layers
#for layer in base_model.layers:
#    print(layer, layer.trainable)



#
# Setup the Data generators
#
#train_datagen = ImageDataGenerator(
#      rescale=1./255,
#      rotation_range=20,
#      width_shift_range=0.2,
#      height_shift_range=0.2,
#      horizontal_flip=True,
#      fill_mode='nearest')
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
train_data = train_datagen.flow_from_directory(
        train_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical')
for image_batch,label_batch in train_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break
N_labels = train_data.labels.max() +1

#
# Create a new model
#
model = models.Sequential()         # Create the model
model.add(base_model)
 
# Add new layers
model.add(layers.Flatten())
#model.add(layers.Dense(1024, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(N_labels, activation='softmax'))
 
# Show a summary of the model. Check the number of trainable parameters
model.summary()
print("[INFO] used mem: {}% - ".format(psutil.virtual_memory().percent) )

 
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
        test_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
N_test_labels = test_data.labels.max() +1
if N_test_labels != N_labels:
    print("Error {0} test labels != {1} training labels".format(N_test_labels, N_labels))


#
# Train the model
#
# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
print("[INFO] model compiled, used mem: {}% - ".format(psutil.virtual_memory().percent) )

# Train the model
start = time.time()
fit_history = model.fit_generator(
      train_data,
      steps_per_epoch=train_data.samples/batch_size,
      epochs=5,
      validation_data=test_data,
      validation_steps=test_data.samples/batch_size,
      verbose=1)
end = time.time()
delta = end - start
print ("[INFO] training completed in {0:.3f} seconds".format(delta))
print("[INFO] model trained, used mem: {}% - ".format(psutil.virtual_memory().percent) )
 
# Save the model
p = os.path.join(out_path, 'xfer_train.h5')
model.save(p)
print("Model saved to " +p)

# Keras objects can't be pickled (thred RLock errors), so copy everything, and
# clear out the .model member
fh_ser = fit_history
fh_ser.model = None

p = os.path.join(out_path, 'xfer_train_hist.pickle')
with open(p, "wb") as fp:
    pickle.dump(fh_ser, fp)
print("Training history saved to " +p)

#
# Check Performance
#
acc = fit_history.history['acc']
val_acc = fit_history.history['val_acc']
loss = fit_history.history['loss']
val_loss = fit_history.history['val_loss']
 
epochs = range(len(acc))
 
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
 
plt.figure()
 
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
 
plt.show()
