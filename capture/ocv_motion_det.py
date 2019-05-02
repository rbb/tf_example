#!/usr/bin/python

# Code from:
# https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# https://www.pyimagesearch.com/2015/06/01/home-surveillance-and-motion-detection-with-the-raspberry-pi-python-and-opencv/

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import io


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", default='ocv_motion_det_conf.json',
        help="path to the JSON configuration file: %(default)s")
ap.add_argument("-v", "--verbose", action='store_true', default=False,
        help="Turn on debug messages")
ap.add_argument("--show_video", action='store_true', default=False,
        help="Turn on video")
ap.add_argument("--dropbox", action='store_true', default=False,
        help="Store frames to dropbox")
ap.add_argument("--dropbox_base_path", default='ocv_motion_det/',
        help="path in dropbox folder for where to store frames: %(default)s")
ap.add_argument("--min_upload_seconds", default=3.0,
        help="TODO: %(default)s")
ap.add_argument("--min_motion_frames", default=4.0,
        help="TODO: %(default)s")
ap.add_argument("--camera_warmup_time", default=1.0,
        help="TODO: %(default)s seconds")
ap.add_argument("--delta_thresh", default=5,
        help="TODO: %(default)s")
ap.add_argument("--min_area", default=5000,
        help="TODO: %(default)s")

# Camera args
ap.add_argument("--fps", default=16,
        help="TODO: %(default)s")
ap.add_argument("--rotation", default=180,
        help="TODO: %(default)s")
ap.add_argument("--exposure_mode", default=None,
        help="One of [off,auto,night,nightpreview,backlight,spotlight,sports,snow,beach,verylong,fixedfps,antishake,fireworks] default: %(default)s")
ap.add_argument("--resolution", default=[640,480], nargs=2,
        help="TODO: %(default)s")

args = ap.parse_args()

# filter warnings, load the configuration and initialize the Dropbox client
warnings.filterwarnings("ignore")

print("opening config: " +args.conf)
#conf = json.load(open(args.conf))
conf = json.load(io.open(args.conf, 'r', encoding='utf-8-sig'))
client = None

# check to see if the Dropbox should be used
if args.dropbox:
    from pyimagesearch.tempimage import TempImage
    import dropbox
    # connect to dropbox and start the session authorization process
    client = dropbox.Dropbox(conf["dropbox_access_token"])
    print("[SUCCESS] dropbox account linked")

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.rotation = args.rotation
camera.resolution = tuple(args.resolution)
camera.framerate = args.fps
if args.exposure_mode:
    camera.exposure_mode = args.exposure_mode
rawCapture = PiRGBArray(camera, size=tuple(args.resolution))
 
# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up for " +str(args.camera_warmup_time) +" seconds")
time.sleep(args.camera_warmup_time)
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

# capture frames from the camera
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image and initialize
    # the timestamp and occupied/unoccupied text
    frame = f.array
    timestamp = datetime.datetime.now()
    text = "Unoccupied"

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        rawCapture.truncate(0)
        print("[INFO] done capturing background.")
        continue
 
    #print("[INFO] after continue.")

    # accumulate the weighted average between the current frame and
    # previous frames, then compute the difference between the current
    # frame and running average
    cv2.accumulateWeighted(gray, avg, 0.5)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

    # threshold the delta image, dilate the thresholded image to fill
    # in holes, then find contours on thresholded image
    thresh = cv2.threshold(frameDelta, args.delta_thresh, 255,
           cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
           cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    n_cnts =0
    for c in cnts:
        print("[INFO] n_cnts = " +str(n_cnts))
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args.min_area:
            continue

        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Occupied"

        # draw the text and timestamp on the frame
        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
           0.35, (0, 0, 255), 1)

        # check to see if the room is occupied
        if text == "Occupied":
          print("[INFO] Occupied.")
          # check to see if enough time has passed between uploads
          if (timestamp - lastUploaded).seconds >= args.min_upload_seconds:
              # increment the motion counter
             motionCounter += 1

             # check to see if the number of frames with consistent motion is
             # high enough
             if motionCounter >= args.min_motion_frames:
                 # check to see if dropbox sohuld be used
                if args.dropbox:
                    # write the image to temporary file
                   t = TempImage()
                   cv2.imwrite(t.path, frame)

                   # upload the image to Dropbox and cleanup the tempory image
                   print("[UPLOAD] {}".format(ts))
                   path = "/{base_path}/{timestamp}.jpg".format(
                           base_path=args.dropbox_base_path, timestamp=ts)
                   client.files_upload(open(t.path, "rb").read(), path)
                   t.cleanup()
                #else:
                # TODO: cv2.imwrite(t.path, frame)

                # update the last uploaded timestamp and reset the motion
                # counter
                lastUploaded = timestamp
                motionCounter = 0

        # otherwise, the room is not occupied
        else:
            motionCounter = 0

    # check to see if the frames should be displayed to screen
    if args.show_video:
        # display the security feed
        cv2.imshow("Security Feed", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)


