import cv2
import numpy as np
import time

# Performance settings
PROCESS_EVERY_N_FRAMES = 5  # Only process every Nth frame
DETECTION_SIZE = (320, 320)  # Smaller input size for faster detection

# Load YOLOv3-tiny
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")

# Use CPU backend as default - more compatible
print("Using CPU backend")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Uncomment these lines if you have proper CUDA support and want to use it
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# print("Using CUDA backend")

# Load COCO class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
# Fix for different OpenCV versions
output_layers_indices = net.getUnconnectedOutLayers()
# Convert indices to Python list if needed
if isinstance(output_layers_indices, np.ndarray):
    output_layers_indices = output_layers_indices.flatten()
    
# Get output layer names
output_layers = []
for i in output_layers_indices:
    # OpenCV may return 0-based or 1-based indices depending on version
    try:
        output_layers.append(layer_names[i - 1])  # 1-based indexing
    except IndexError:
        output_layers.append(layer_names[i])      # 0-based indexing

# Start webcam - try to set lower resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize variables for FPS calculation
frame_count = 0
start_time = time.time()
fps = 0

# Store last detection results
last_boxes = []
last_confidences = []
last_class_ids = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Get original dimensions
    height, width = frame.shape[:2]
    
    # Process only every N frames for better performance
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        # Resize for processing (smaller size = faster)
        input_frame = cv2.resize(frame, DETECTION_SIZE)
        
        # Blob from frame
        blob = cv2.dnn.blobFromImage(input_frame, 1/255.0, DETECTION_SIZE, swapRB=True, crop=False)
        net.setInput(blob)
        
        # Run detection
        outputs = net.forward(output_layers)
        
        # Clear previous results
        last_boxes = []
        last_confidences = []
        last_class_ids = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > 0.5 and classes[class_id] in ["car", "truck", "bus", "motorbike"]:
                    # YOLO returns normalized coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w_box = int(detection[2] * width)
                    h_box = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w_box / 2)
                    y = int(center_y - h_box / 2)
                    
                    last_boxes.append([x, y, w_box, h_box])
                    last_confidences.append(float(confidence))
                    last_class_ids.append(class_id)
        
        # Non-max suppression to reduce overlapping boxes
        if last_boxes:  # Only perform NMS if we have detected objects
            indexes = cv2.dnn.NMSBoxes(last_boxes, last_confidences, 0.5, 0.4)
            
            # Reset lists to keep only NMS results
            temp_boxes = []
            temp_confidences = []
            temp_class_ids = []
            
            # Handle OpenCV versions difference in return type of NMSBoxes
            if len(indexes) > 0:  # Make sure we have detections after NMS
                if isinstance(indexes, tuple):
                    # For newer OpenCV versions that return a tuple
                    indexes = indexes[0] if indexes else []
                
                # Handle both numpy array and direct list
                if isinstance(indexes, np.ndarray):
                    indexes = indexes.flatten()
                    
                for i in indexes:
                    temp_boxes.append(last_boxes[i])
                    temp_confidences.append(last_confidences[i])
                    temp_class_ids.append(last_class_ids[i])
                    
            # Update with NMS results
            last_boxes = temp_boxes
            last_confidences = temp_confidences
            last_class_ids = temp_class_ids
    
    # Always draw the most recent detection results
    for i in range(len(last_boxes)):
        x, y, w_box, h_box = last_boxes[i]
        label = f"{classes[last_class_ids[i]]} {last_confidences[i]:.2f}"
        color = (0, 255, 0)
        
        # Ensure box coordinates are within frame boundaries
        x = max(0, x)
        y = max(0, y)
        x_end = min(width, x + w_box)
        y_end = min(height, y + h_box)
        
        cv2.rectangle(frame, (x, y), (x_end, y_end), color, 2)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Calculate and display FPS
    frame_count += 1
    if frame_count % 10 == 0:
        end_time = time.time()
        fps = 10 / (end_time - start_time)
        start_time = end_time
    
    # Display FPS on frame
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show frame
    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()