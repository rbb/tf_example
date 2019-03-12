#!/bin/bash

#pip3 uninstall mxnet
#pip uninstall tensorflow tensorboard Keras-Applications Keras-Preprocessing tensorflow_hub --yes

#pip3 uninstall tensorflow --yes 
#pip3 uninstall tensorboard Keras-Applications Keras-Preprocessing tensorflow_hub 
#pip3 uninstall numpy scipy--yes
#rm -rf $HOME/.local/lib/python3.6/site-packages/tensorflow/

cd /home/russell/projects/tfpi/tensorflow/tensorflow_pkg/
pip3 install --upgrade --no-cache tensorflow*.whl
cd -
pip3 install tensorboard Keras-Applications Keras-Preprocessing tensorflow_hub scipy

# vim tw=0
