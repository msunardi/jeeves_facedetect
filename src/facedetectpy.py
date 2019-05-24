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
from std_msgs.msg import String, Float32
from sensor_msgs.msg import CompressedImage, Image

__author__ =  'Simon Haller <simon.haller at uibk.ac.at>'
__version__=  '0.1'
__license__ = 'BSD'
# Python libs
import sys, time

# numpy and scipy
from scipy.ndimage import filters

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

class FaceDetect(threading.Thread):
    def __init__(self, cascade, nested):
        self.image_pub = rospy.Publisher("/output/image_raw/compressed", Image, queue_size=1)
        self.position_pub = rospy.Publisher("/output/position", String, queue_size=1)
        self.yaw_pub = rospy.Publisher("/head/cmd_pose_yaw", Float32, queue_size=1)
        self.pitch_pub = rospy.Publisher("/head/cmd_pose_pitch", Float32, queue_size=1)
        # subscribed Topic
        self.subscriber = rospy.Subscriber("/usb_cam/image_raw/compressed",
            CompressedImage, self.callback, queue_size=1)
        if VERBOSE :
            print("subscribed to /camera/image/compressed")

        self.cascade = cascade
        self.nested = nested

        self.bridge = CvBridge()

        threading.Thread.__init__(self)
        self.sleeper = rospy.Rate(10)

    def callback(self, data):
        # img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # img = data.data
        np_arr = np.fromstring(data.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # img = self.bridge.imgmsg_to_cv2(img, "bgr8")
        # img = np.array(data.data)
        # gray = cv2.imread(img, 0)
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
            rospy.loginfo(rects[0])
            rect = rects[0]
            width = 640
            height = 480
            facex = rect[0] + (rect[2] - rect[0])/2.0 - width/2.0
            facey = rect[1] + (rect[3] - rect[1])/2.0 - height/2.0
            self.position_pub.publish("x: {}, y: {}".format(facex, facey))
            self.yaw_pub.publish(facex)
            self.pitch_pub.publish(facey)
        # #### Create CompressedIamge ####
        # msg = Image()
        # msg.header.stamp = rospy.Time.now()
        # # msg.format = "jpeg"
        # msg.data = np.array(cv2.imencode('.jpg', vis)[1]).tostring()
        # # Publish new image
        # self.image_pub.publish(msg)

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

    cam = create_capture(video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

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
