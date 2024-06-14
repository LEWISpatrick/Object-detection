import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model_dir = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model'
try:
    detect_fn = tf.saved_model.load(model_dir)
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

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

# Function to process frames
def process_frame(frame):
    # Convert frame to tensor
    input_tensor = tf.convert_to_tensor(np.expand_dims(frame, 0), dtype=tf.uint8)
    detections = detect_fn(input_tensor)

    # Extract detection boxes, classes, and scores
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    height, width, _ = frame.shape
    for i in range(len(detection_boxes)):
        if detection_scores[i] > 0.5:  # Only consider high-confidence detections
            box = detection_boxes[i]
            ymin, xmin, ymax, xmax = box
            start_point = (int(xmin * width), int(ymin * height))
            end_point = (int(xmax * width), int(ymax * height))
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)

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
