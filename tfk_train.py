#!/usr/bin/python3

# From https://gogul09.github.io/software/flower-recognition-deep-learning

# organize imports
from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import io
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import psutil
import time

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
features_path = ld_conf_path("features_file")
labels_path   = ld_conf_path("labels_file")
results_path  = config["results_file"]
model_path    = ld_conf_path("model_file")
classifier_path = ld_conf_path("classifier_file")
num_classes    = config["num_classes"]
seed           = config["seed"]
test_size     = config["test_size"]

# import features and labels
h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')
print ("[INFO] open h5py, used mem: {}% - ".format(psutil.virtual_memory().percent))

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)
print ("[INFO] copy to np, used mem: {}% - ".format(psutil.virtual_memory().percent))

h5f_data.close()
h5f_label.close()
print ("[INFO] close h5py, used mem: {}% - ".format(psutil.virtual_memory().percent))

del features_string
del labels_string
print ("[INFO] del h5py, used mem: {}% - ".format(psutil.virtual_memory().percent))

# verify the shape of features and labels
print ("[INFO] features shape: {}".format(features.shape))
print ("[INFO] labels shape: {}".format(labels.shape))

print ("[INFO] training started...")
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(features,
                                                                  labels,
                                                                  test_size=test_size,
                                                                  random_state=seed)

print ("[INFO] splitted train and test data...")
print ("[INFO] train data  : {}".format(trainData.shape))
print ("[INFO] test data   : {}".format(testData.shape))
print ("[INFO] train labels: {}".format(trainLabels.shape))
print ("[INFO] test labels : {}".format(testLabels.shape))
print ("[INFO] used mem    : {}% - ".format(psutil.virtual_memory().percent))

del features
del labels
print ("[INFO] del features, labels; used mem: {}% - ".format(psutil.virtual_memory().percent))

# use logistic regression as the model
print ("[INFO] creating model...")
model = LogisticRegression(random_state=seed)
model.fit(trainData, trainLabels)
print ("[INFO] created model, used mem: {}% - ".format(psutil.virtual_memory().percent))


# use rank-1 and rank-5 predictions
print ("[INFO] evaluating model...")
rank_1 = 0
rank_5 = 0

# loop over test data
n = 0
Nmod = 50
mod_start = time.time()
for (label, features) in zip(testLabels, testData):
    # predict the probability of each class label and
    # take the top-5 class labels
    predictions = model.predict_proba(np.atleast_2d(features))[0]
    predictions = np.argsort(predictions)[::-1][:5]

    # rank-1 prediction increment
    if label == predictions[0]:
        rank_1 += 1

    # rank-5 prediction increment
    if label in predictions:
        rank_5 += 1

    if n % Nmod == 0:
        mod_end = time.time()
        mod_delta = mod_end - mod_start
        mod_avg_time = mod_delta / Nmod
        print ("[INFO] feature {} - ".format(n) 
            +"used mem: {}% - ".format(psutil.virtual_memory().percent)
            +"Avg Time: {0:3.4f}".format(mod_avg_time) )
        mod_start = time.time()
    n += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100
rank_5 = (rank_5 / float(len(testLabels))) * 100

print("Rank-1: {:.2f}%".format(rank_1))
print("Rank-5: {:.2f}%".format(rank_5))

# write the accuracies to file
f = open(results_path, "w")
f.write("Rank-1: {:.2f}%\n".format(rank_1))
f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

# evaluate the model of test data
preds = model.predict(testData)

# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# dump classifier to file
print ("[INFO] saving model...")
pickle.dump(model, open(classifier_path, 'wb'))

# display the confusion matrix
print ("[INFO] confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_path)))

# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm, annot=True, cmap="gray")   # Set2, jet, gnuplot2
plt.show()

