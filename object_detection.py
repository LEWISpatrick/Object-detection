import cv2
import numpy as np
import tensorflow as tf
import pygame
from pygame import mixer
import time

# Initialize pygame mixer
mixer.init()

# Load the sound file
sound = mixer.Sound('./GET-OUT.mp3')

# Load the pre-trained model
model_dir = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
try:
    detect_fn = tf.saved_model.load(model_dir)
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# COCO dataset class names
class_names = [
    'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
    'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window',
    'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'toilet paper', 'hand'
]

def list_cameras():
    index = 0
    camera_indices = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            camera_indices.append(index)
        cap.release()
        index += 1
    return camera_indices

def select_camera(preferred_index=1):
    camera_indices = list_cameras()
    print("Available camera indices:", camera_indices)
    if not camera_indices:
        print("No cameras found.")
        return -1
    if preferred_index in camera_indices:
        return preferred_index

    for index in camera_indices:
        cap = cv2.VideoCapture(index)
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {index}", frame)
            print(f"Press 'y' to select camera {index}, or any other key to try the next camera.")
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord('y'):
                return index
        cap.release()
    return -1

# Select the correct camera, starting with index 1 as preferred
camera_index = select_camera(preferred_index=1)
if camera_index == -1:
    print("No suitable camera found")
    exit()

cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()

# Variable to store the time when a person was last detected
last_detection_time = None

# Function to process frames
def process_frame(frame):
    global last_detection_time

    # Convert frame to tensor
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)
    detections = detect_fn(input_tensor)

    # Extract detection boxes, classes, and scores
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    height, width, _ = frame.shape
    person_detected = False
    for i in range(len(detection_boxes)):
        if detection_scores[i] > 0.5:  # Only consider high-confidence detections
            box = detection_boxes[i]
            ymin, xmin, ymax, xmax = box
            start_point = (int(xmin * width), int(ymin * height))
            end_point = (int(xmax * width), int(ymax * height))
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

            # Get class name
            class_id = detection_classes[i]
            class_name = class_names[class_id]

            # Put class name text on the frame
            label = f"{class_name}: {detection_scores[i]:.2f}"
            cv2.putText(frame, label, (start_point[0], start_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if the detected object is a person
            if class_name == 'person':
                person_detected = True

    if person_detected:
        # If a person is detected, play the sound if it has been more than 3 seconds since the last detection
        if last_detection_time is None or time.time() - last_detection_time > 3:
            sound.play()
            last_detection_time = time.time()

    return frame

# Capture and process frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    processed_frame = process_frame(frame)

    # Display the resulting frame
    cv2.imshow('Object Detection', processed_frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
