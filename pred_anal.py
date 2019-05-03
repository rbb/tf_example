#!/usr/bin/python3

# From https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/

# organize imports
#from __future__ import print_function
import numpy as np
#import h5py
import os
import io
import json
import time
import pickle

import tensorflow as tf
#import keras

# Gather imagenet labels into groups that match our training sets
class label_types(object):
    def __init__(self, class_indices=None):
        self.d = {
            # imagenet classes/labels
            "cat": np.array([281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293]),
            "dog": np.concatenate(np.array(range(151,269)), 275)
            #"person": np.concatenate(np.array(range(151,269)), 275)
        }
        self.rev_map = {} 
        if class_indices:
            self.class_indices = class_indices
            self.map_labels()
        else:
            self.class_indices = {}
    def map_labels(self, class_indices=None):
        if not class_indices:
            class_indices = self.class_indices
        elif not self.class_indices:
            self.class_indices = class_indices
            
        for k,v in class_indices.items():
            self.rev_map[v] = k
                
    def match(self, prediction, label_ind):
        N = len(prediction)
        matches = np.zeros(N)
        for n in range(N):
            p = prediction[n]
            #print("len(p) = {}".format(len(p)))
            l = int(label_ind[n])
            #print("match n,p,l =" +str((n,p,l)))
            meta_label_list = self.d[self.rev_map[l]] 
            if p.shape == ():
                retval = np.any(meta_label_list == p)
            else:
                retval = 0
                for m in range(p.shape[0]):
                    if np.any(meta_label_list == p[m]):
                        retval = 1
            #print("match {}= ".format(n) +str(retval))
            if retval:
                matches[n] = 1
        return matches
    def accuracy(self, prediction, label_ind):
        matches = self.match(prediction, label_ind)
        #print("pred_anal.label_types: matches = ", str(matches))
        #print("pred_anal.label_types: len(prediction) = ", str(len(prediction)))
        fraction = np.sum(matches) / float(len(prediction))
        return fraction

def get_imagenet_labels(path='ImageNetLabels.txt', url='https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'):
    imagenet_labels_path = tf.keras.utils.get_file(path, url)
    imagenet_labels = np.array(open(labels_path).read().splitlines())

def result_top(x):
    return np.argmax(x, axis=-1)

def result_top_k(x, k):
    return np.argpartition(x, -k)[:,-k:]

#def print_failures():
#    print('Failures:')
#    for n in range(cats.shape[0]):
#        if cats[n] == 0.0:
#            print('Example # ' +str(n) +' classified as ' +str(imagenet_labels[result_top[n]]))
 

def score(top_pred, meta_labels):
    N = len(top_pred)
    print("pred_anal.score: N = ", str(N))
    correct = 0
    for n in range(N):
        for key in meta_labels.keys:
            if np.any(top_pred(n) == label_types[key]):
                correct += 1
    accuracy = correct / float(N)
    return accuracy

