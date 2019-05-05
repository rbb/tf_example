#!/bin/bash

# based on post at:
# https://www.pyimagesearch.com/2015/02/23/install-opencv-and-python-on-your-raspberry-pi-2-and-b/

if command -v aptitude &>/dev/null; then
    pm=aptitude
elif command -v apt &>/dev/null; then
    pm=apt
elif command -v apt-get &>/dev/null; then
    pm=apt-get
else
    echo "package manager not found"
    exit
fi

# Install the GTK development library. This library is used to build GUIs and is
# required for the highgui library of OpenCV which allows you to view images on
# your screen:
sudo $pm install libgtk2.0-dev

# Install the necessary video I/O packages. These packages are used to load video
# files using OpenCV
#sudo $pm install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
sudo $pm install libjpeg-dev libtiff-dev libjasper-dev libpng-dev

# Install libraries that are used to optimize various operations within OpenCV:
sudo $pm install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev

sudo $pm install python-pip python-numpy python-dev python-opencv

pip install imutils

# Not OCV, but helps with using the Mac to connect - so the pi's home dir can be
# mounted in Mac Os.
sudo $pm install netatalk
