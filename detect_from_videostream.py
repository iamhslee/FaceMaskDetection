# Face Mask Detection - detect_from_videostream.py
# 2020. 09. 01. by Hyunseo Lee

# Usage
# python detect_from_videostream.py

# Import necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# FMD info
print("\n================================================")
print("\nFace Mask Detection - detect_from_videostream.py")
print("\n2020. 09. 01. by Hyunseo Lee")
print("\n================================================\n")

def detect_and_predict_mask(frame, faceNet, maskNet):
	# Grab the dimensions of the frame and then construct a blob from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# Pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# Initialize our list of faces, their corresponding locations, and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# Loop over the detections
	for i in range(0, detections.shape[2]):
		# Extract the confidence (i.e., probability) associated with the detection
		confidence = detections[0, 0, i, 2]

		# Filter out weak detections by ensuring the confidence is greater than the minimum confidence
		if confidence > args["confidence"]:
			# Compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# Ensure the bounding boxes fall within the dimensions of the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# Extract the face ROI, convert it from BGR to RGB channel ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# Add the face and bounding boxes to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# Only make a predictions if at least one face was detected
	if len(faces) > 0:
		# For faster inference we'll make batch predictions on all faces at the same time rather than one-by-one predictions in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# Return a 2-tuple of the face locations and their corresponding locations
	return (locs, preds)

# Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str, default="CAFFE_DNN", help="Face Detector Model Path")
ap.add_argument("-m", "--model", type=str, default="face_mask_detection.model", help="Face Mask Detector Model Path")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load face detection model
print("[FMD] [INFO] Loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"], "face_detection.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load face mask detection model
print("[FMD] [INFO] Loading face mask detector model...")
maskNet = load_model(args["model"])

# Initialize the video stream
print("[FMD] [INFO] Starting video stream...")
vs = VideoStream(src=0).start()

# Help info to quit cv2 video stream window
print("\n==============================================")
print("\nTo quit cv2 video stream window, Press key [0]")
print("\n==============================================")

# Loop over the frames from the video stream
while True:
	# Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# Detect faces in the frame and determine if they are wearing a face mask or not
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# Loop over the detected face locations and their corresponding locations
	for (box, pred) in zip(locs, preds):
		# Unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# Determine the class label and color we'll use to draw the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# Display the label and bounding box rectangle on the output frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# Show the output frame
	cv2.imshow("Face Mask Detection - VideoSteam", frame)
	key = cv2.waitKey(1) & 0xFF

	# If the `0` key was pressed, break from the loop
	if key == ord("0"):
		break

# Cleanup
cv2.destroyAllWindows()
vs.stop()

# Detection done
print("\n================================================")
print("\nFace Mask Detection - detect_from_videostream.py")
print("\nDetection complete.")
print("\n================================================")