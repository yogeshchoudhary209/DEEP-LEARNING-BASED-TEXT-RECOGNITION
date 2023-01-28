#!/tmp/ENTER/envs/tesseract/bin/python
# coding=utf-8

import imutils
from imutils.object_detection import non_max_suppression
import rospy
import numpy as np
import argparse
import cv2
import glob
import os
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/tmp/ENTER/envs/tesseract/bin/tesseract'


def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical data used to derive potential bounding box coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)

ap = argparse.ArgumentParser()
ap.add_argument("-east", "--east", type=str, required=True,
	help="path to input EAST text detector")
ap.add_argument("-v", "--video", type=str,
	help="path to optinal input video file")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())

layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
net = cv2.dnn.readNet(args["east"])
bridge = CvBridge()
msg = None

def img_callback(rec_img):
    global msg
    rospy.loginfo(rospy.get_caller_id() + "I heard something")
    msg=rec_img


def TextDetector():
	(newW, newH) = (args["width"], args["height"])
	pub = rospy.Publisher("/usb_cam/output", Image, queue_size = 1)
	rospy.init_node("text_detector_subscribe", anonymous=True)
	rospy.Subscriber("/usb_cam/infer", Image, img_callback)
	rate = rospy.Rate(10)
	while not rospy.is_shutdown():
		if msg is None:
			continue
		frame = bridge.imgmsg_to_cv2(msg, "bgr8")

		(W, H) = (None, None)
		(rW, rH) = (None, None)
		if W is None or H is None:
			(H, W) = frame.shape[:2]
			rW = W / float(newW)
			rH = H / float(newH)
		
		# construct a blob from the frame and then perform a forward pass
		# of the model to obtain the two output layer sets
		blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)
		net.setInput(blob)
		(scores, geometry) = net.forward(layerNames)
		(rects, confidences) = decode_predictions(scores, geometry)
		boxes = non_max_suppression(np.array(rects), probs=confidences)
		results = []
		for (startX, startY, endX, endY) in boxes:
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)
			
			dX = int((endX - startX) * args["padding"])
			dY = int((endY - startY) * args["padding"])
			
			# apply padding to each side of the bounding box, respectively
			startX = max(0, startX - dX)
			startY = max(0, startY - dY)
			endX = min(W, endX + (dX * 2))
			endY = min(H, endY + (dY * 2))
			
			roi = frame[startY:endY, startX:endX]
			config = ("-l eng --oem 1 --psm 7")
			text = pytesseract.image_to_string(roi, lang='eng', config=config)
			results.append(((startX, startY, endX, endY), text))
		results = sorted(results, key=lambda r:r[0][1])
		for ((startX, startY, endX, endY), text) in results:
			text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
			cv2.remap
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
			cv2.putText(frame, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
		cv_image = pub.publish(bridge.cv2_to_imgmsg(frame, "bgr8"))
		
		cv_image = cv2.resize(frame, (newW, newH))
		
		# publish final results
		cv2.imshow("Image window", cv_image)
		
		if cv2.waitKey(30) & 0xFF == ord('q'):
			break

if __name__ == '__main__':
	TextDetector()



