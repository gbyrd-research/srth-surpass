import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
import os
import json
import time
import sys
import matplotlib.pyplot as plt
from PIL import Image as img
import random
import copy

import argparse


class ImageSaver:
    def __init__(self, filename):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/jhu_daVinci/left/image_raw", Image, self.callback)
        self.save_path = os.path.expanduser(f'~/chole_ws/data/atracsys_data/{filename}')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.image_count = 0

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image_name = os.path.join(self.save_path, f"image_{self.image_count:04d}.jpg")
            cv2.imwrite(image_name, cv_image)
            rospy.loginfo(f"Saved image {image_name}")
            self.image_count += 1
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert image: {e}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Save ROS messages to CSV files.')
    parser.add_argument('--file', type=str, default='mark_r_2', help='Base name for the output CSV files')
    args = parser.parse_args()

    rospy.init_node('image_saver', anonymous=True)
    image_saver = ImageSaver(args.file)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
