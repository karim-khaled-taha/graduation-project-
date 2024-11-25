import cv2
from datetime import datetime
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from load_encoding_images import SimpleFacerec  # Import the class
import os

# Initialize FaceMeshDetector and SimpleFacerec class
detector = FaceMeshDetector()
sfr = SimpleFacerec()

# Define the encoding file path
encoding_file_path = "face_encodings.pkl"

# Load face encodings from the "face_encodings.pkl" file initially
sfr.load_saved_encodings(encoding_file_path)

# Define directory paths
save_dir = "recognized_face"
database_dir = "DataBase"

# Create directories if they do not exist
os.makedirs(save_dir, exist_ok=True)
os.makedirs(database_dir, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Camera not found or cannot be opened.")
    exit()

while True:
    # Read frame from the camera
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Display the live camera feed with instructions
    cv2.putText(frame, "Press Enter to Capture and Recognize Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, "Press 'n' to Capture New Photo for Database", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(frame, "Press 's' to load encoding file", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow('Live Camera Feed', frame)

    key = cv2.waitKey(1)

    # Save face encodings when 's' key is pressed and remove old encodings
    if key == ord('s'):  
        sfr.clear_encodings()
        sfr.load_encoding_images(database_dir)
        sfr.save_encodings(encoding_file_path)
        sfr.load_saved_encodings(encoding_file_path)
        # # Automatically calculate ranges for all classes (persons)
        sfr.calculate_ranges_for_all_classes()
        print("New encodings saved and calculated ranges, old encodings removed, and loaded successfully.")

    # Capture new photo for database when 'n' key is pressed
    if key == ord('n'):
        person_name = input("Enter the name of the person: ").strip()
        if person_name:
            new_person_image = frame.copy()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
             # Create a folder for the person if it doesn't exist
            person_folder = os.path.join(database_dir, person_name)
            os.makedirs(person_folder, exist_ok=True)
            
            # Save the new image with a timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_person_filename = os.path.join(person_folder, f"{timestamp}.jpg")
            new_person_image = frame.copy()
            cv2.imwrite(new_person_filename, new_person_image)
            print(f"New person image saved to {new_person_filename}")
        else:
            print("No name entered. Photo not saved.")

    # Recognize the face when Enter is pressed
    if key == 13:  # Enter key
        snapshot = frame.copy()
        snapshot, faces = detector.findFaceMesh(snapshot, draw=False)
        
        if faces:
            # Iterate over each detected face
            for face in faces:
                # Select specific points on the face to calculate the distance
                pointLeft = face[145]
                pointRight = face[374]
                cv2.line(snapshot, pointLeft, pointRight, (0, 200, 0), 3)
                cv2.circle(snapshot, pointLeft, 5, (255, 0, 255), cv2.FILLED)
                cv2.circle(snapshot, pointRight, 5, (255, 0, 255), cv2.FILLED)
                w, _ = detector.findDistance(pointLeft, pointRight)

                # Assuming W (real distance between two points) and f (focal length)
                W = 11.4  # Known distance in cm between landmarks
                f = 700   # Focal length, adjust according to your setup
                d = (W * f) / w  # Calculate depth
                
                # Display depth for each face
                cvzone.putTextRect(snapshot, f'Depth: {int(d)}cm', (face[10][0] - 100, face[10][1] - 50), scale=2)

        # Detect and label faces
        face_locations, face_names = sfr.detect_known_faces(snapshot)
        if d < 100:
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(snapshot, (left, top), (right, bottom), color, 2)
                cv2.putText(snapshot, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        # Show snapshot with annotations
        cv2.imshow('Captured Frame with Face Recognition', snapshot)
        
        # Save the snapshot with recognized faces and distances
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(save_dir, f"recognized_face_{timestamp}.jpg")
        cv2.imwrite(filename, snapshot)
        print(f"Image saved as {filename}")


    # Break the loop when the 'q' key is pressed
    if key == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
