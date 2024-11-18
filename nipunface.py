import cv2
import os

# Directory containing saved photos
photo_dir = r"stored_faces"

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load photos and create a dictionary of names and photos
def load_photos():
    photo_data = {}
    
    for file in os.listdir(photo_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(file)[0]  # Use the file name (without extension) as the person's name
            file_path = os.path.join(photo_dir, file)
            
            # Read and store the image
            image = cv2.imread(file_path)
            photo_data[name] = image
    
    return photo_data

# Compare detected faces with stored images
def match_face(input_face, photo_data):
    for name, photo in photo_data.items():
        # Convert both input and stored photos to grayscale
        photo_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        input_face_gray = cv2.cvtColor(input_face, cv2.COLOR_BGR2GRAY)
        
        # Resize stored photo to match the input face size for comparison
        resized_photo = cv2.resize(photo_gray, (input_face_gray.shape[1], input_face_gray.shape[0]))
        
        # Compute similarity score using template matching
        result = cv2.matchTemplate(input_face_gray, resized_photo, cv2.TM_CCOEFF_NORMED)
        similarity = result.max()
        
        if similarity > 0.5:  # Threshold for a match
            return name

    return "Unknown"

# Detect faces and recognize them
def detect_and_recognize():
    # Load stored photos
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
        
        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = frame[y:y+h, x:x+w]
            
            # Recognize the face
            name = match_face(face_roi, photo_data)
            
            # Draw a rectangle around the face and display the name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
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
