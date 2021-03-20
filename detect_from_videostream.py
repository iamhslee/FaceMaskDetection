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
import cvlib as cv
import csv
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# FMD info
print("\n================================================")
print("\nFace Mask Detection - detect_from_videostream.py")
print("\n2020. 09. 01. by Hyunseo Lee")
print("\n================================================\n")

# Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default="face_mask_detection.model", help="Face Mask Detector Model Path")
args = vars(ap.parse_args())

# CSV file path
videostreamCSV = open('./CSV/VideoStream/videostream.csv', 'w', encoding='utf-8')
videostreamCSVwriter = csv.writer(videostreamCSV)

# Load face mask detection model
print("[FMD] [INFO] Loading face mask detector model...")
model = load_model(args["model"])

# Initialize the video stream
print("[FMD] [INFO] Starting video stream...")
vs = VideoStream(src=0).start()

# Help info to quit cv2 video stream window
print("\n==============================================")
print("\nTo quit cv2 video stream window, Press key [0]")
print("\n==============================================")

# Variables about faces number & The presense or absence about wearing mask
facesNum = int(0)
withMaskNum = int(0)
withoutMaskNum = int(0)

# Loop over the frames from the video stream
while True:
	# Grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=480)

	(h, w) = frame.shape[:2]

	faces = []

	faces, confidence = cv.detect_face(frame)

	# Loop over the detected face locations and their corresponding locations
	for face in faces:
		(startX, startY) = face[0], face[1]
		(endX, endY) = face[2], face[3]
		face = frame[startY:endY, startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)
		(mask, withoutMask) = model.predict(face)[0]
		label = "Mask" if mask > withoutMask else "No Mask"

		# Count faces number & Increse number when people wear mask or not
		if label == "Mask":
			withMaskNum += 1
			facesNum += 1
		
		if label == "No Mask":
			withoutMaskNum += 1
			facesNum += 1
		
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
		label = "{}: {:.3f}%".format(label, max(mask, withoutMask) * 100)
		cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.putText(frame, "Number of Faces : " + str(facesNum), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(frame, "Faces with Mask : " + str(withMaskNum), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.putText(frame, "Faces without Mask : " + str(withoutMaskNum), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		videostreamCSVwriter.writerow([facesNum, withMaskNum, withoutMaskNum])

		# Reset variables after print text one time
		facesNum = int(0)
		withMaskNum = int(0)
		withoutMaskNum = int(0)

	# Show the output frame
	cv2.imshow("Face Mask Detection - VideoSteam", frame)
	key = cv2.waitKey(1) & 0xFF

	# If the `0` key was pressed, break from the loop
	if key == ord("0"):
		break

# Cleanup
cv2.destroyAllWindows()
vs.stop()
videostreamCSV.close()

# Detection done
print("\n================================================")
print("\nFace Mask Detection - detect_from_videostream.py")
print("\nDetection complete.")
print("\n================================================")