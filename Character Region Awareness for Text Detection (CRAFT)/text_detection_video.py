#!/usr/bin/env python
# coding=utf-8

"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import argparse
import cv2
from skimage import io
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

import imutils
import rospy
import numpy as np
import argparse
import cv2
import glob
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from collections import OrderedDict
def copyStateDict(state_dict):
	if list(state_dict.keys())[0].startswith("module"):
		start_idx = 1
	else:
		start_idx = 0
	new_state_dict = OrderedDict()
	for k, v in state_dict.items():
		name = ".".join(k.split(".")[start_idx:])
		new_state_dict[name] = v
	return new_state_dict

def str2bool(v):
	return v.lower() in ("yes", "y", "true", "t", "1")
    
parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--video', type=str, help='path to optinal input video file')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', type=str, help='pretrained refiner model')


args = parser.parse_args()

def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, args.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes, polys, ret_score_text


vs = cv2.VideoCapture("/home/inspirelab/Documents/Text-Based-Signage-Recognition-master/video.mp4")
print(vs.isOpened())
bridge = CvBridge()


def TextDetector():
	# load net
	net = CRAFT()     
	print('Loading weights from checkpoint (' + args.trained_model + ')')
	if args.cuda:
		net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
	else:
		net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

	if args.cuda:
		net = net.cuda()
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = False

	net.eval()

	# LinkRefiner
	refine_net = None
	if args.refine:
		from refinenet import RefineNet
		refine_net = RefineNet()
		print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
		if args.cuda:
			refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
			refine_net = refine_net.cuda()
			refine_net = torch.nn.DataParallel(refine_net)
		else:
			refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

		refine_net.eval()
		args.poly = True
	

	pub = rospy.Publisher('/usb_cam/inference', Image, queue_size = 1)
	rospy.init_node("text_detector", anonymous=True)
	
	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		ret, frame = vs.read()
		if not ret:
			break
		
		msg = bridge.cv2_to_imgmsg(frame, "bgr8")
		#pub.publish(msg)
		rospy.Subscriber("/usb_cam", Image, msg)
		frame = bridge.imgmsg_to_cv2(msg, "bgr8")
	
		orig = frame.copy()
		
		bboxes, polys, score_text = test_net(net, orig, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
		texts=None
		for i, box in enumerate(bboxes):
			poly = np.array(box).astype(np.int32).reshape((-1))
			strResult = ','.join([str(p) for p in poly]) + '\r\n'
			

			poly = poly.reshape(-1, 2)
			cv2.polylines(orig, [poly.reshape((-1, 1, 2))], True, color=(0, 0, 255), thickness=2)
			ptColor = (0, 255, 255)
		
		cv_image = pub.publish(bridge.cv2_to_imgmsg(orig, "bgr8"))
		(newW, newH) = (320, 320)
		
		# cv_image = cv2.resize(orig, (newW, newH))
		# cv2.imshow("Image window", cv_image)
		# if cv2.waitKey(30) & 0xFF == ord('q'):
			# break

if __name__ == '__main__':
	TextDetector()




