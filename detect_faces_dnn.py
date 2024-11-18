import cv2
import numpy as np
import os

# Directory containing saved photos
photo_dir = "stored_faces"

# Paths to DNN model files (download these beforehand)
prototxt_path = "deploy.prototxt"  # Model configuration
model_path = "res10_300x300_ssd_iter_140000.caffemodel"  # Pre-trained model weights

# Load the DNN model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Load photos and create a dictionary of names and photos
def load_photos():
    photo_data = {}
    for file in os.listdir(photo_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(file)[0]  # Use the file name (without extension) as the person's name
            file_path = os.path.join(photo_dir, file)
            image = cv2.imread(file_path)
            photo_data[name] = image
    return photo_data

# Compare detected face with stored photos using template matching
def match_face(input_face, photo_data):
    for name, photo in photo_data.items():
        # Convert both input and stored photos to grayscale
        photo_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        input_face_gray = cv2.cvtColor(input_face, cv2.COLOR_BGR2GRAY)
        
        # Resize stored photo to match the input face size
        resized_photo = cv2.resize(photo_gray, (input_face_gray.shape[1], input_face_gray.shape[0]))
        
        # Compute similarity score using template matching
        result = cv2.matchTemplate(input_face_gray, resized_photo, cv2.TM_CCOEFF_NORMED)
        similarity = result.max()
        
        if similarity > 0.4:  # Threshold for a match
            return name
    return "Unknown"

# Detect faces using OpenCV DNN
def detect_faces_dnn(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            faces.append((x, y, x1-x, y1-y))
    return faces

# Detect and recognize faces in the webcam feed
def detect_and_recognize():
    photo_data = load_photos()
    if not photo_data:
        print("No photos found in the database. Please add photos to the 'stored_faces' directory.")
        return
    
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces in the frame
        faces = detect_faces_dnn(frame)
        
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Recognize the face
            name = match_face(face_roi, photo_data)
            
            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Face Detection and Recognition", frame)
        
        # Quit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main Program
if __name__ == "__main__":
    print("Detecting and recognizing faces...")
    detect_and_recognize()
