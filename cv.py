import cv2
import cvzone
from ultralytics import YOLO

# Path to the video file
video_path = r'C:\Users\Akshay\Downloads\8774702-hd_1920_1080_30fps.mp4'

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Load the YOLO face detection model
try:
    facemodel = YOLO('yolov8n-face.pt')
except Exception as e:
    print(f"Error: Could not load YOLO model. {e}")
    exit()

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    # Resize the frame for consistent display
    frame = cv2.resize(frame, (700, 500))

    # Perform face detection
    face_results = facemodel.predict(frame, conf=0.40)
    
    # Draw bounding boxes around detected faces
    for face_result in face_results:
        for box in face_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, [x1, y1, w, h], l=9, rt=3)

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
