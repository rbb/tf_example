#!/usr/bin/python

from picamera import PiCamera
from time import sleep

camera = PiCamera()
camera.rotation = 180

camera.start_preview()
sleep(1)
camera.capture('image.jpg')
camera.stop_preview()

#camera.start_preview()
#camera.start_recording('video.h264')
#sleep(10)
#camera.stop_recording()
#camera.stop_preview()
