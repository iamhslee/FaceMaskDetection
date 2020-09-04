# Face Mask Detection - detect_from_image.py
# 2020. 09. 01. by Hyunseo Lee

# Usage
# python detect_from_image.py --image <Image Path>

# Import necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import cvlib as cv
import os

# Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image Path")
ap.add_argument("-m", "--model", type=str, default="face_mask_detection.model", help="Face Mask Detector Model Path")
args = vars(ap.parse_args())

# FMD info
print("\n==========================================")
print("\nFace Mask Detection - detect_from_image.py")
print("\n2020. 09. 01. by Hyunseo Lee")
print("\n==========================================\n")

# Load face mask detection model
print("[FMD] [INFO] Loading face mask detector model...")
model = load_model(args["model"])

# Load the input image from disk, clone it, and grab the image spatial dimensions
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]

# Pass the blob through the network and obtain the face detections
print("[FMD] [INFO] Computing face detections...")
faces, confidence = cv.detect_face(image)

# Variables about faces number & The presense or absence about wearing mask
facesNum = int(0)
withMaskNum = int(0)
withoutMaskNum = int(0)

# Loop over the detections
for face in faces:
	(startX, startY) = face[0], face[1]
	(endX, endY) = face[2], face[3]
	face = image[startY:endY, startX:endX]
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
	cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
	cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	cv2.putText(image, "Number of Faces : " + str(facesNum), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(image, "Faces with Mask : " + str(withMaskNum), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
	cv2.putText(image, "Faces without Mask : " + str(withoutMaskNum), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Help info to quit cv2 image window
print("\n=======================================")
print("\nTo quit cv2 image window, Press key [0]")
print("\n=======================================")

cv2.imshow("Face Mask Detection - Image", image)
cv2.waitKey(0)

# Detection done
print("\n==========================================")
print("\nFace Mask Detection - detect_from_image.py")
print("\nDetection complete.")
print("\n==========================================")