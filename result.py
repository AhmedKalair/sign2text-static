import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from utils.get_sentence import get_detections_array
from utils.get_sentence import get_sentence

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("keras_model.h5", "labels.txt") 
labels = open("labels.txt", "r").readlines()
labels = [label.strip().split(' ')[1] for label in labels]
# labels = ['call', 'friend', 'house', 'chicken_burger', 'i_love_you', 'love', 'more', 'one', 'what', 'you']

offset = 20
imgWidth = 600
imgHeight = 400

previous_prediction = None
consecutive_detections = 0
threshold = 4
detections = set()
last_detection_time = time.time()

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    imgWhite = np.ones((imgHeight, imgWidth, 3), np.uint8) * 255

    if hands:
        # print("There are hands in the frame")
        aspectRatio = None
        if len(hands) >= 2:
            hand_boxes = [hand['bbox'] for hand in hands]
            x_min = min(box[0] for box in hand_boxes)
            y_min = min(box[1] for box in hand_boxes)
            x_max = max(box[0] + box[2] for box in hand_boxes)
            y_max = max(box[1] + box[3] for box in hand_boxes)

            imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset]
            imgCropShape = imgCrop.shape

            aspectRatio = (y_max - y_min) / (x_max - x_min)

        elif len(hands) == 1:
            x_min, y_min, w, h = hands[0]['bbox']
            x_max, y_max = x_min + w, y_min + h
            imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset]
            imgCropShape = imgCrop.shape

            aspectRatio = h / w

        if not imgCrop.size == 0:
            if aspectRatio > 1:
                h_start = (imgHeight - imgCropShape[0]) // 2
                h_end = h_start + imgCropShape[0]
                w_start = (imgWidth - imgCropShape[1]) // 2
                w_end = w_start + imgCropShape[1]

                if h_start >= 0 and h_end <= imgHeight and w_start >= 0 and w_end <= imgWidth:
                    imgWhite[h_start:h_end, w_start:w_end] = imgCrop
            else:
                h_start = (imgHeight - imgCropShape[0]) // 2
                h_end = h_start + imgCropShape[0]
                w_start = (imgWidth - imgCropShape[1]) // 2
                w_end = w_start + imgCropShape[1]

                if h_start >= 0 and h_end <= imgHeight and w_start >= 0 and w_end <= imgWidth:
                    imgWhite[h_start:h_end, w_start:w_end] = imgCrop

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgWhite)

            if prediction[index] > 0.90:
                cv2.putText(imgOutput, labels[index], (x_min, y_min -20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)
                print(prediction, index)

                # use gpt here
                previous_prediction, consecutive_detections, detections = get_detections_array(labels[index], previous_prediction, consecutive_detections, detections, threshold)
                # print(previous_prediction, consecutive_detections, detections)

                # Update last detection time
                last_detection_time = time.time()

    # Check if 3 seconds have passed without detection and if detections set/array has more than 2 elements
    if time.time() - last_detection_time >= 3 and len(detections) >= 2:
        print("Sending detections to GPT")
        get_sentence(detections)

        # Clear variables of detections
        detections.clear()
        previous_prediction = None
        consecutive_detections = 0
        

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)