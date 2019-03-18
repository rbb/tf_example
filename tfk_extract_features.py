#!/usr/bin/python3

# From https://gogul09.github.io/software/flower-recognition-deep-learning


# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
#import cv2
import h5py
import os
import io
import json
import datetime
import time
import psutil

# load the user configs
#with open('tfk_conf.json') as f:    
#  config = json.load(f)
config = json.load(io.open('tfk_conf.json', 'r', encoding='utf-8-sig'))

def ld_conf_path(f, config=config, opath="out_path"):
    p = os.path.join(config[opath], config[f])
    return p

# config variables
model_name    = config["model"]
weights       = config["weights"]
include_top   = config["include_top"]
train_path    = config["train_path"]
out_path      = config["out_path"]
features_path = ld_conf_path("features_file")
labels_path   = ld_conf_path("labels_file")
results       = config["results_file"]
model_path    = ld_conf_path("model_file")
test_size     = config["test_size"]

# start time
print ("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not
if model_name == "vgg16":
    base_model = VGG16(weights=weights)
    model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
    image_size = (224, 224)
elif model_name == "vgg19":
    base_model = VGG19(weights=weights)
    model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
    image_size = (224, 224)
elif model_name == "resnet50":
    base_model = ResNet50(weights=weights)
    model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
    image_size = (224, 224)
elif model_name == "inceptionv3":
    base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
    model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
    image_size = (299, 299)
elif model_name == "inceptionresnetv2":
    base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
    model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
    image_size = (299, 299)
elif model_name == "mobilenet":
    base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
    #model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
    #model = Model(input=base_model.input, output=base_model.get_layer('conv_pw_13_relu').output)
    model = Model(input=base_model.input, output=base_model.get_layer(index=-1).output)
    
    #base_model.layers.pop()
    #new_model = Sequential()
    #new_model.add(base_model)
    #model.layers[-1].outbound_nodes = []
    #new_model.add(Dense(num_class, activation='softmax'))
    image_size = (224, 224)
elif model_name == "xception":
    base_model = Xception(weights=weights)
    model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
    image_size = (299, 299)
else:
    base_model = None

# Freeze the first layers because these contain very basic feature detection
for layer in model.layers[:10]:
    layer.trainable = False

print ("[INFO] successfully loaded base model and model...")

# path to training dataset
train_labels = os.listdir(train_path)

# encode the labels
print ("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in train_labels])



# Init output dir, and delete old (HD5) files
os.makedirs(out_path, exist_ok=True)
if os.path.exists(features_path):
    print("[STATUS] Found " +features_path +", deleting it")
    os.remove(features_path)
if os.path.exists(labels_path):
    print("[STATUS] Found " +labels_path +", deleting it")
    os.remove(labels_path)
    
# Find total number of files, to determine the HD5 array number of columns
Nfiles = 0
for k, label in enumerate(train_labels):
    cur_path = train_path + "/" + label
    paths = glob.glob(cur_path + "/*.jpg")
    Nlabel = len(paths)
    Nfiles += Nlabel
print("Nfiles = " +str(Nfiles))

# Get the feature size, to determine the HD5 array number of rows
img = image.load_img(paths[0], target_size=image_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
feature = model.predict(x)
flat = feature.flatten()
print("flat.shape = " +str(flat.shape))

h5f_data = h5py.File(features_path, 'w')
#h5f_data.create_dataset('dataset_1', data=np.array(features))
h5f_data.create_dataset('dataset_1',
        shape=(Nfiles,flat.shape[0]),
        compression=None,
        chunks=True )
        #chunks=(100, feature.shape[0]) )

# loop over all the labels in the folder
labels   = []
K = len(train_labels)
n = 0
Nmod = 50
for k, label in enumerate(train_labels):
    cur_path = train_path + "/" + label
    label_count = 1
    start = time.time()
    mod_start = time.time()
    paths = glob.glob(cur_path + "/*.jpg")
    Nlabel = len(paths)
    for image_path in paths:
        img = image.load_img(image_path, target_size=image_size)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = model.predict(x)
        flat = feature.flatten()
        h5f_data['dataset_1'][n,:] = flat
        labels.append(label)
        if label_count % Nmod == 0:
            mod_end = time.time()
            mod_delta = mod_end - mod_start
            mod_avg_time = mod_delta / Nmod
            print ("[INFO] processed - " 
                +label +" - {0:3.2f}% - ".format(100.0*k/K)
                +"{0} - {1:3.2f}% - ".format(label_count, 100.0*label_count/Nlabel)
                +"used mem: {}% - ".format(psutil.virtual_memory().percent)
                +"Avg Time: {0:3.4f}".format(mod_avg_time) )
            mod_start = time.time()
        label_count += 1
        n += 1
    end = time.time()
    delta = end - start
    avg_time = delta / Nlabel
    print ("[INFO] completed label - " +label +" - avg time " +str(avg_time))

# encode the labels using LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# get the shape of training labels
print ("[STATUS] training labels: {}".format(le_labels))
print ("[STATUS] training labels shape: {}".format(le_labels.shape))

# save features and labels
h5f_data.close()

# TODO: labels are variable length, but can we chuck the data into the
# h5f_label? If so, then maybe use a single file instead of two.
h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels)) 
h5f_label.close()
print ("[STATUS] features and labels saved..")

# save model and weights
model_json = model.to_json()
with open(model_path + str(test_size) + ".json", "w") as json_file:
  json_file.write(model_json)

# save weights
model.save_weights(model_path + str(test_size) + ".h5")
print("[STATUS] saved model and weights to disk..")


# end time
end = time.time()
print ("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))

