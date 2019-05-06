#!/usr/bin/python

# Code from:
# https://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
# https://www.pyimagesearch.com/2015/06/01/home-surveillance-and-motion-detection-with-the-raspberry-pi-python-and-opencv/

# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import warnings
import datetime
import imutils
import json
import time
import cv2
import io
import os
import argparse


# Learn about API authentication here: https://plot.ly/python/getting-started
# Find your api_key here: https://plot.ly/settings/api
import plotly.plotly as py
import plotly.graph_objs as pgo


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", metavar='STR', default='ocv_motion_det_conf.json',
        help="path to the JSON configuration file: %(default)s")
ap.add_argument("--show_video", action='store_true', default=False,
        help="Turn on video")
ap.add_argument('--dropbox', default=False,
        type=lambda x: (str(x).lower() == 'true'),
        metavar = 'true/false',
        help="Store frames to dropbox. Default: %(default)s")
ap.add_argument('--plotly', default=False,
        type=lambda x: (str(x).lower() == 'true'),
        metavar = 'true/false',
        help="Update plotly graph. Default: %(default)s")
ap.add_argument("--dropbox_path", metavar='STR', default='ocv_motion_det/',
        help="path in dropbox folder for where to store frames: %(default)s")
ap.add_argument('--local', dest='store_local', default=False,
        type=lambda x: (str(x).lower() == 'true'),
        metavar = 'true/false',
        help="Store frames locally. Default: %(default)s")

# Alg args
gp_alg = ap.add_argument_group('Algorithm Args')
gp_alg.add_argument("--min_out_seconds", metavar='N', default=2.0,
        help="Minimum seconds between storing frames (local or dropbox): %(default)s")
gp_alg.add_argument("--min_motion_frames", metavar='N', default=1,
        help="Minimum number of frames with motion before looking for contours: %(default)s")
gp_alg.add_argument("--camera_warmup", metavar='N', default=0.5,
        help="Seconds to wait for start: %(default)s")
gp_alg.add_argument("--delta_thresh", metavar='N', default=5,
        help="TODO: %(default)s")
gp_alg.add_argument("--min_area", metavar='N', default=5000,
        help="Minimum area that has to change to call it a contour: %(default)s")
gp_alg.add_argument("--out_dir", metavar='STR', default="./frames/",
        help="Where to store output frames (those deemed occupied): %(default)s")

# logging args
gp_log = ap.add_argument_group('Logging Args')
gp_log.add_argument('--log', default=True,
        type=lambda x: (str(x).lower() == 'true'),
        metavar = 'true/false',
        dest = 'log_en',
        help="Update (local) log of brightness values. Default: %(default)s")
gp_log.add_argument("--log_dir", default=".",
        help="Where to store output log files: %(default)s")
gp_log.add_argument("--log_interval", metavar='N', default="60",
        help="Seconds between log entries: %(default)s")

# Camera args
gp_cam = ap.add_argument_group('Camera Args')
gp_cam.add_argument("--fps", default=2,
        help="Video frames per second: %(default)s")
gp_cam.add_argument("--rotation", metavar='N', default=180,
        help="Video rotation: %(default)s")
gp_cam.add_argument("--exposure_mode", metavar='STR', default=None,
        help="One of [off,auto,night,nightpreview,backlight,spotlight,sports,snow,beach,verylong,fixedfps,antishake,fireworks] default: %(default)s")
gp_cam.add_argument("--resolution", metavar='N', default=[640,480], nargs=2,
        help="Video resolution: %(default)s")

ap.add_argument("-v", "--verbose", action='store_true', default=False,
        help="Turn on debug messages")
args = ap.parse_args()

class Avgs():
    def __init__(self):
        self.sum = 0.0
        self.n = 1.0
    def add(self, val):
        self.sum += val;
        self.n += 1.0
    def clear(self):
        self.sum = 0.0;
        self.n = 1.0
    def avg(self, clear=False):
        avg = self.sum / self.n
        if clear:
            self.clear()
        return avg
#    def print_status(self):
#        print(clearelf.sum = 0;
#        self.n = 1

def update_plotly(fname, x, y, auto_open=False):
    new_data = pgo.Scatter(x=[x], y=[y] )
    data = pgo.Data( [ new_data ] )
    plot_url = py.plot(data,
        filename = fname,
        fileopt = 'extend',
        auto_open = auto_open)
    return plot_url

# filter warnings, load the configuration and initialize the Dropbox client
warnings.filterwarnings("ignore")

print("opening config: " +args.conf)
#conf = json.load(open(args.conf))
conf = json.load(io.open(args.conf, 'r', encoding='utf-8-sig'))
client = None

if args.log_en:
    flog = open(os.path.join(args.log_dir,
        'ocv_motion_det_avg_brightness.csv'),'a+')

# check to see if the Dropbox should be used
if args.dropbox:
    #from pyimagesearch.tempimage import TempImage
    from tempimage import TempImage
    import dropbox
    # connect to dropbox and start the session authorization process
    client = dropbox.Dropbox(conf["dropbox_access_token"])
    print("[SUCCESS] dropbox account linked")

if not os.path.isdir(args.out_dir):
    print("[INFO] creating directory " +str(args.out_dir))
    os.makedirs(args.out_dir)

# initialize the camera and grab a reference to the raw camera capture
camera = PiCamera()
camera.rotation = args.rotation
camera.resolution = tuple(args.resolution)
camera.framerate = args.fps
print("[INFO] camera fps = " +str(camera.framerate))
if args.exposure_mode:
    print("[INFO] PiCamera.EXPOSURE_MODES = " +str(PiCamera.EXPOSURE_MODES))
    print("[INFO] exposure_mode = " +str(camera.exposure_mode))
    print("[INFO] attempting to set exposer_mode = " +str(args.exposure_mode))
    camera.exposure_mode = str(args.exposure_mode)
    print("[INFO] exposure_mode = " +str(camera.exposure_mode))

    # TODO: try manually toggling the LED, which is connected to the IR cut
    # https://github.com/BigNerd95/CameraLED
    # https://github.com/ArduCAM/RPI_Motorized_IRCut_Control
    # or try precompiled binary: http://www.arducam.com/downloads/modules/RaspberryPi_camera/piCamLed.zip

    # TODO: Try toggling through /sys/
    # disable_camera_led=1 
    # in config.txt: the following commands run as root user will switch the led on (or IR cut filter in depending on your camera). 
    #
    # $>echo 32 > /sys/class/gpio/export 
    # $>echo out > /sys/class/gpio/gpio32/direction 
    # $>echo 1 > /sys/class/gpio/gpio32/value 
    #
    # To switch back out again; 
    # $>echo 0 > /sys/class/gpio/gpio32/value see less 

rawCapture = PiRGBArray(camera, size=tuple(args.resolution))
 
# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up for " +str(args.camera_warmup) +" seconds")
time.sleep(args.camera_warmup)
avg = None
lastUploaded = datetime.datetime.now()
motionCounter = 0

# capture frames from the camera
last_log_time = datetime.datetime.now()
agray = Avgs()
aconts = Avgs()
athresh = Avgs()
amotion = Avgs()
amotion_conts = Avgs()
amotion_area = Avgs()
for f in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image and initialize
    # the timestamp and occupied/unoccupied text
    frame = f.array
    timestamp = datetime.datetime.now()
    fts = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

    # resize the frame, convert it to grayscale, and blur it
    #frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    agray.add( gray.mean() )

    # if the average frame is None, initialize it
    if avg is None:
        print("[INFO] starting background model...")
        avg = gray.copy().astype("float")
        rawCapture.truncate(0)
        print("[INFO] done capturing background.")
        continue
 
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
    athresh.add( thresh.mean() )


    # loop over the contours
    found_contours = False
    not_small_conts = []
    frame_marked = frame.copy()
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) < args.min_area:
            continue

        # Compute the bounding box for the contour, and draw it on the frame
        (x, y, w, h) = cv2.boundingRect(c)
        not_small_conts.append(frame[y:y + h, x:x + w])

        cv2.rectangle(frame_marked, (x, y), (x + w, y + h), (0, 255, 0), 2)
        found_contours = True

        # Draw the timestamp on the frame
        ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
        cv2.putText(frame_marked, ts, (10, frame_marked.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
           0.35, (0, 0, 255), 1)


    # check to see if the room is occupied
    if found_contours:
        print("[INFO] " +ts +" found " +str(len(not_small_conts))) +" contours in frame."
        #print("[INFO] Occupied.")

        # increment the motion counter
        motionCounter += 1

        # check to see if enough time has passed between uploads
        if (timestamp - lastUploaded).seconds >= args.min_out_seconds:

            # check to see if the number of frames with consistent motion is
            # high enough
            if motionCounter >= args.min_motion_frames:
                print("[INFO] found motion")
                amotion.add(1.0)
                amotion_conts.add(len(not_small_conts))
                for c in not_small_conts:
                    amotion_area.add( c.shape[0]*c.shape[1] )
                # check to see if dropbox sohuld be used
                if args.dropbox:
                    # write the image to temporary file
                    t = TempImage()
                    cv2.imwrite(t.path, frame_marked)

                    # upload the image to Dropbox and cleanup the tempory image
                    print("[UPLOAD] {}".format(ts))
                    path = "/{base_path}/{timestamp}.jpg".format(
                           base_path=args.dropbox_path, timestamp=ts)
                    client.files_upload(open(t.path, "rb").read(), path)
                    t.cleanup()
                if args.store_local:
                    fn = "{out_dir}/{timestamp}_frame.jpg".format(
                           out_dir=args.out_dir, timestamp=fts)
                    print("[INFO] writing frame " +str(fn))
                    cv2.imwrite(fn, frame_marked)
                    n=1
                    for c in not_small_conts:
                        cn = "{out_dir}/{timestamp}_cont-{n}.jpg".format(
                           out_dir=args.out_dir, timestamp=fts, n=n)
                        print("[INFO] writing contour " +str(cn))
                        cv2.imwrite(cn, c)
                        n = n +1

            # update the last uploaded timestamp and reset the motion
            # counter
            lastUploaded = timestamp

        motionCounter = 0

    # otherwise, the room is not occupied
    else:
        motionCounter = 0

    # Logging
    aconts.add( len(not_small_conts))
    dt = (timestamp - last_log_time).total_seconds()
    if dt > float(args.log_interval):
        row_data = [ agray.avg(), athresh.avg(), aconts.avg(),
                amotion.avg(), amotion_conts.avg(), amotion_area.avg()]
        row_str = ', '.join([str(i) for i in row_data])
        print("[INFO] log data: " +row_str)
        last_log_time = datetime.datetime.now()
        if args.log_en:
            lts = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            flog.write(lts +', ' +str(agray.avg()) +', ' +row_str +'\n')
            print("[INFO] updated log")
        if args.plotly:
            br_url = update_plotly('ocv_motion_det_avg_brightness', last_log_time,
                    agray.avg())
            br_url = update_plotly('ocv_motion_det_avg_thresh', last_log_time,
                    athresh.avg())
            br_url = update_plotly('ocv_motion_det_avg_cnts', last_log_time,
                    aconts.avg())
            br_url = update_plotly('ocv_motion_det_avg_motion', last_log_time,
                    amotion.avg())
            print("[INFO] plotly url: " +str(br_url))
        agray.clear()
        athresh.clear()
        aconts.clear()
        amotion.clear()
        amotion_conts.clear()
        amotion_area.clear()

    # check to see if the frames should be displayed to screen
    if args.show_video:
        # display the security feed
        cv2.imshow("Security Feed", frame_marked)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
            break

    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)


if args.log_en:
    flog.close()
