#!/usr/bin/python

"""
Simply display the contents of the webcam with optional mirroring using OpenCV 
via the new Pythonic cv2 interface.  Press <esc> to quit.
"""

import cv2


def show_webcam(mirror=False):
    cam = cv2.VideoCapture(0)
    #cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
    while True:
        ret_val, img = cam.read()
        if mirror: 
            img = cv2.flip(img, 1)
        cv2.imshow('my webcam', img)
        key = cv2.waitKey(1)
        if key == 27: 
            break  # esc to quit
        if key == ord('q'): 
            break  # q to quit
    cv2.destroyAllWindows()


def main():
    show_webcam(mirror=True)


if __name__ == '__main__':
    main()
