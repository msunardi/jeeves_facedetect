#!/usr/bin/env python

'''
face detection using haar cascades
Stolen from opencv samples
USAGE:
    facedetect.py [--cascade <cascade_fn>] [--nested-cascade <cascade_fn>] [<video_source>]
'''
# REF: http://wiki.ros.org/rospy_tutorials/Tutorials/WritingImagePublisherSubscriber
# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
# import cv2
import threading
# import rospy
from std_msgs.msg import String, Float32, Float64
from sensor_msgs.msg import CompressedImage, Image

__author__ =  'Simon Haller <simon.haller at uibk.ac.at>'
__version__=  '0.1'
__license__ = 'BSD'
# Python libs
import sys, time
from math import *
import random

# numpy and scipy
from scipy.ndimage import filters
from scipy.stats import norm
# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

VERBOSE=False
# # local modules
from video import create_capture
from common import clock, draw_str

import rospkg

# get an instance of RosPack with the default search paths
rospack = rospkg.RosPack()

# get the file path for rospy_tutorials
path = rospack.get_path('facedetectpy')

from cv_bridge import CvBridge, CvBridgeError

IMG_WIDTH=640
IMG_HEIGHT=480

class FaceDetect(threading.Thread):
    def __init__(self, cascade, nested):
        self.image_pub = rospy.Publisher("/output/image_raw", Image, queue_size=1)
        self.position_pub = rospy.Publisher("/output/position", String, queue_size=1)
        #self.yaw_pub = rospy.Publisher("/head/cmd_pose_yaw", Float32, queue_size=1)
        self.yaw_pub = rospy.Publisher("/head_pan_joint/command", Float64, queue_size=1)
        self.pitch_pub = rospy.Publisher("/head/cmd_pose_pitch", Float32, queue_size=1)
        self.eye_yaw_pub = rospy.Publisher("/eye/yaw", Float32, queue_size=1)
        self.eye_pitch_pub = rospy.Publisher("/eye/pitch", Float32, queue_size=1)
        # subscribed Topic
        #self.subscriber = rospy.Subscriber("/usb_cam/image_raw/compressed",
        #   CompressedImage, self.callback, queue_size=1)
        self.subscriber = rospy.Subscriber("/usb_cam/image_raw",
            Image, self.callback, queue_size=1)
        
        if VERBOSE :
            print("subscribed to /camera/image/compressed")

        self.cascade = cascade
        self.nested = nested

        self.bridge = CvBridge()

        threading.Thread.__init__(self)
        self.sleeper = rospy.Rate(10)

    def callback(self, data):
        img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # img = data.data
        #np_arr = np.fromstring(data.data, np.uint8)
        #img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        # img = np.array(data.data)
        # gray = cv2.imread(img, 0)
        #try:
        #    img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        #except CvBridgeError as e:
        #    print( e)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # t = clock()
        rects = self.detect(gray, self.cascade)
        vis = img.copy()
        self.draw_rects(vis, rects, (0, 255, 0))
        if not self.nested.empty():
            for x1, y1, x2, y2 in rects:
                roi = gray[y1:y2, x1:x2]
                vis_roi = vis[y1:y2, x1:x2]
                subrects = self.detect(roi.copy(), self.nested)
                self.draw_rects(vis_roi, subrects, (255, 0, 0))
        # dt = clock() - t

        # draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
        # cv2.imshow('facedetect', vis)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))
        # rospy.loginfo(type(rects))
        if isinstance(rects, np.ndarray):
            rospy.loginfo("{}".format(rects))
            rect = rects[0]
            width = IMG_WIDTH
            height = IMG_HEIGHT

            # find facebox horizontal offset from center
            facex = rect[0] + (rect[2] - rect[0])/2.0
            eyes_yaw = self.normalize_eye_yaw(width - facex, width) # x axis of eyes is inverted
            #facex_center = facex - width/2.0
            facex_center = self.normalize_head_yaw(rect, width)
            # find facebox vertical offset from center
            facey = rect[1] + (rect[3] - rect[1])/2.0
            eyes_pitch = self.normalize_eye_pitch(facey, height)
            #facey_center = -facey + height/2.0

            #self.position_pub.publish("x: {}, y: {}".format(facex_center, facey_center))
            self.yaw_pub.publish(Float64(facex_center))
            #self.pitch_pub.publish(facey_center)
            self.position_pub.publish("eye_x: {}, eye_y: {}".format(eyes_yaw, eyes_pitch))
            self.eye_yaw_pub.publish(eyes_yaw)
            self.eye_pitch_pub.publish(eyes_pitch)
        # #### Create CompressedIamge ####
        # msg = Image()
        # msg.header.stamp = rospy.Time.now()
        # # msg.format = "jpeg"
        # msg.data = np.array(cv2.imencode('.jpg', vis)[1]).tostring()
        # # Publish new image
        # self.image_pub.publish(msg)

    def normalize_head_yaw(self, rect, ref):
        r0 = rect[0]
        r2 = rect[2]
        x_center = ref/2
        rx = (r2 + r0)/2.0 # Horizontal center of rect
        
        # Use normal distribution to control activation of neck movements (-5 - +5)
        # How much to move
        x = (rx - x_center)/x_center * 5.0
        # How much activation for the move
        p = 0.4 - norm.pdf(x)
        pos = p * x * 1.5 # 5.0/2
        rospy.loginfo("x/p/pos: {}/{}/{}".format(x, p, pos))
        return pos

    # def normalize_head_pitch(self, rect, ref):

    def normalize_eye_pitch(self, value, ref):
        # Normalize values for eye position
        # must consider camera placement and adjust pitch value
        segments = 6
        segment_size = float(ref)/segments
        out = value/segment_size
        return max(min(ceil(out), 5.0), 3.0)

    def normalize_eye_yaw(self, value, ref):
        # Normalize values for eye position
        segments = 6
        segment_size = float(ref)/segments
        out = value/segment_size
        return max(min(ceil(out), 4.0), 2.0)

    def detect(self, img, cascade):
        rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:,2:] += rects[:,:2]
        return rects

    def draw_rects(self, img, rects, color):
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    def run(self):
        while not rospy.is_shutdown():
            # ret, img = self.cam.read()
            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # gray = cv2.equalizeHist(gray)
            #
            # t = clock()
            # rects = detect(gray, self.cascade)
            # vis = img.copy()
            # draw_rects(vis, rects, (0, 255, 0))
            # if not self.nested.empty():
            #     for x1, y1, x2, y2 in rects:
            #         roi = gray[y1:y2, x1:x2]
            #         vis_roi = vis[y1:y2, x1:x2]
            #         subrects = detect(roi.copy(), self.nested)
            #         draw_rects(vis_roi, subrects, (255, 0, 0))
            # dt = clock() - t
            #
            # draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
            # cv2.imshow('facedetect', vis)
            #
            # if cv2.waitKey(5) == 27:
            #     break
            self.sleeper.sleep()
        cv2.destroyAllWindows()

def main(args):
    import sys, getopt
    print(__doc__)

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    # cascade_fn = args.get('--cascade', "data/haarcascades/haarcascade_frontalface_alt.xml")
    # nested_fn  = args.get('--nested-cascade', "../data/haarcascades/haarcascade_eye.xml")
    cascade_fn = path + '/data/haarcascade_frontalface_alt.xml'
    nested_fn = path + '/data/haarcascade_eye.xml'
    print(cascade_fn)
    print(nested_fn)

    cascade = cv2.CascadeClassifier(cascade_fn)
    nested = cv2.CascadeClassifier(nested_fn)

    #cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

    fd = FaceDetect(cascade, nested)
    fd.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

# def main(args):
#     '''Initializes and cleanup ros node'''
#     ic = FaceDetect()
#     rospy.init_node('image_feature', anonymous=True)
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         print("Shutting down ROS Image feature detector module")
#     cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('facedetectpy', anonymous=True)
    main(sys.argv)
# def detect(img, cascade):
#     rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
#                                      flags=cv2.CASCADE_SCALE_IMAGE)
#     if len(rects) == 0:
#         return []
#     rects[:,2:] += rects[:,:2]
#     return rects
#
# def draw_rects(img, rects, color):
#     for x1, y1, x2, y2 in rects:
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#
# if __name__ == '__main__':
#     import sys, getopt
#     print(__doc__)
#
#     args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
#     try:
#         video_src = video_src[0]
#     except:
#         video_src = 0
#     args = dict(args)
#     cascade_fn = args.get('--cascade', "../data/haarcascades/haarcascade_frontalface_alt.xml")
#     nested_fn  = args.get('--nested-cascade', "../data/haarcascades/haarcascade_eye.xml")
#
#     cascade = cv2.CascadeClassifier(cascade_fn)
#     nested = cv2.CascadeClassifier(nested_fn)
#
#     cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')
#
#     while True:
#         ret, img = cam.read()
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         gray = cv2.equalizeHist(gray)
#
#         t = clock()
#         rects = detect(gray, cascade)
#         vis = img.copy()
#         draw_rects(vis, rects, (0, 255, 0))
#         if not nested.empty():
#             for x1, y1, x2, y2 in rects:
#                 roi = gray[y1:y2, x1:x2]
#                 vis_roi = vis[y1:y2, x1:x2]
#                 subrects = detect(roi.copy(), nested)
#                 draw_rects(vis_roi, subrects, (255, 0, 0))
#         dt = clock() - t
#
#         draw_str(vis, (20, 20), 'time: %.1f ms' % (dt*1000))
#         cv2.imshow('facedetect', vis)
#
#         if cv2.waitKey(5) == 27:
#             break
#     cv2.destroyAllWindows()
#
#
#
# 	def run(self):
# 		#self.redis_sub.subscribe(**{'response': self.redis_handler})
# 		while not rospy.is_shutdown():
# 			#self.redis_sub.get_message()
# 			self.sleeper.sleep()
#
# def main():
# 	rospy.init_node('num_finger', anonymous=True)
# 	finger_math = Num_Finger()
# 	finger_math.start()
# 	rospy.spin()
#
# if __name__ == '__main__':
# 	main()
