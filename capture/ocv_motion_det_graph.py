#!/usr/bin/python

import datetime
import io
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

#from os.path import expanduser
#home = expanduser("~")


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-f', "--log_file", default="ocv_motion_det_avg_brightness.csv",
        help="log file as input (csv): %(default)s")


ap.add_argument("-v", "--verbose", action='store_true', default=False,
        help="Turn on debug messages")
args = ap.parse_args()

print("opening log file: " +args.log_file)
df = pd.read_csv(args.log_file)

# brightness, threshold, n_contours, motion, n_motion_conts, motion_area

#----------------------------------
fig1 =plt.figure(1)
fig1.clf()
ax1 = plt.subplot(6,1,1)
#----------------------------------
#h1 = df.brightness.plot(style='go', ax=ax1, markersize=3, legend=False)
h2 = df.plot(x='date',y='brightness',style='go', ax=ax1, markersize=3, legend=False)

#h1.set_ylabel('Brightness')
ax1.set_xticklabels([])
#h1.set_ylim(7.0, 13.0)
#h1.set_yticks([7,8,9,10,11,12,13])
ax1.grid(True)

#----------------------------------
ax2 = plt.subplot(6,1,2)
#----------------------------------
h2 = df.plot(x='date',y='threshold',style='go', ax=ax2, markersize=3, legend=False)

#ax2.set_ylabel('Threshold')
#ax2.set_ylim(5.0, 12.0)
#ax2.set_yticks(range(5,12))
ax2.grid(True)
ax2.set_xticklabels([])

#----------------------------------
ax3 = plt.subplot(6,1,3)
#----------------------------------
h3 = df.n_contours.plot(style='go', ax=ax3, markersize=3, legend=False)

#ax3.set_ylabel('N Contours')
#ax3.set_ylim(5.0, 12.0)
#ax3.set_yticks(range(5,12))
ax3.grid(True)
ax3.set_xticklabels([])

#----------------------------------
ax4 = plt.subplot(6,1,4)
#----------------------------------
h4 = df.motion.plot(style='go', ax=ax4, markersize=3, legend=False)

#ax4.set_ylabel('Motion')
#ax4.set_ylim(5.0, 12.0)
#ax4.set_yticks(range(5,12))
ax4.grid(True)
ax4.set_xticklabels([])

#----------------------------------
ax5 = plt.subplot(6,1,5)
#----------------------------------
h5 = df.n_motion_conts.plot(style='go', ax=ax5, markersize=3, legend=False)

#ax5.set_ylabel('N Motion Contours')
#ax5.set_ylim(5.0, 12.0)
#ax5.set_yticks(range(5,12))
ax5.grid(True)
ax5.set_xticklabels([])

#----------------------------------
ax6 = plt.subplot(6,1,6)
#----------------------------------
#h6 = df.motion_area.plot(style='go', ax=ax6, markersize=3, legend=False)
h6 = df.plot(x='date',y='motion_area',style='go', ax=ax6, markersize=3, legend=False)

#ax6.set_ylabel('Motion Area')
#ax6.set_ylim(5.0, 12.0)
#ax6.set_yticks(range(5,12))
ax6.grid(True)
#ax6.set_xticklabels([])

plt.show()
