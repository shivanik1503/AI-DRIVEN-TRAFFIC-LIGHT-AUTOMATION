import cv2
import numpy as np

import picamera
import RPi.GPIO as GPIO
import time
from time import sleep 


camera = picamera.PiCamera() 
camera.resolution = (1028, 720)

# Load the pre-trained MobileNet SSD model and config file
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Define the class labels (COCO dataset)
class_labels = ['background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
                'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
                'sheep', 'sofa', 'train', 'truck']

# Access the camera (can be replaced with video file)
#cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
#if not cap.isOpened():
#    print("Error: Could not open video stream.")
#    exit()

relay = 5

GPIO.setmode(GPIO.BCM)       # Use BCM GPIO numbers
GPIO.setup(relay, GPIO.OUT) # DB7
GPIO.output(relay, 0) # voice

while True:
    # Read a frame from the camera
    #ret, frame = cap.read()
    
    camera.capture('image.jpg')
    time.sleep(0.1)
    
    frame = cv2.imread('image.jpg')
    
    #if not ret:
    #    print("Error: Failed to capture image.")
    #    break

    # Get the image dimensions
    height, width = frame.shape[:2]

    # Convert the image to a blob for the neural network
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5))

    # Set the blob as input to the neural network
    net.setInput(blob)

    # Get the detections
    detections = net.forward()
    print("----------------------------------")
    count = 0

    # Loop through all detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # confidence score for each detection

        if confidence > 0.2:  # You can adjust this threshold for detection sensitivity
            # Get the class index
            class_index = int(detections[0, 0, i, 1])
            print (class_labels[class_index])
            
            # Only look for "car" class (index 7 for car in the COCO dataset)
            if class_labels[class_index] == "car" or class_labels[class_index] == "bottle" or class_labels[class_index] == "boat":
                print ("car")
                # Get the bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box around the detected car
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # Label the bounding box with "Car"
                label = f"{class_labels[class_index]} : {confidence * 100:.2f}%"
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                count = count + 1
                print ("final count :" +str(count))
                
            if count > 2:       
                GPIO.output(relay, True) # LED
                time.sleep(1)
            else:
                GPIO.output(relay, False) # LED
                time.sleep(0.1)

    # Show the video with the detected cars
    #cv2.imshow("Car Detection - MobileNet SSD", frame)

    # Break the loop if 'q' is pressed
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
