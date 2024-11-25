import face_recognition
import cv2
import os
import glob
import numpy as np
import pickle


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = 0.5
        self.class_ranges = {}  # Dictionary to store ranges for each class

    def load_encoding_images(self, images_path):
        # Load images and encode them
        folder_paths = [os.path.join(images_path, f) for f in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, f))]
        print("{} folders found.".format(len(folder_paths)))

        for folder_path in folder_paths:
            person_name = os.path.basename(folder_path)  # Use folder name as class name
            image_paths = glob.glob(os.path.join(folder_path, "*.*"))  # Load all image files in folder
            print(f"Found {len(image_paths)} images for {person_name}.")

            for img_path in image_paths:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Unable to read image at {img_path}. Skipping...")
                    continue  # Skip if the image cannot be read

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Get encoding
                encodings = face_recognition.face_encodings(rgb_img)

                if encodings:  # Check if any encodings were found
                    img_encoding = encodings[0]
                    self.known_face_encodings.append(img_encoding)
                    self.known_face_names.append(person_name)  # Append folder name as class
                    print(f"Encoded image from {person_name} successfully.")
                else:
                    print(f"No face encodings found for image {img_path}. Skipping...")

        print("Encoding images loaded.")

    def save_encodings(self, filename="face_encodings.pkl"):
        # Save the encodings to a file
        with open(filename, "wb") as file:
            pickle.dump((self.known_face_encodings, self.known_face_names), file)
        print(f"Encodings saved to {filename}")

    def clear_encodings(self):
     self.known_face_encodings = []
     self.known_face_names = []    

    def load_saved_encodings(self, filename="face_encodings.pkl"):
        # Load encodings from a file
        with open(filename, "rb") as file:
            self.known_face_encodings, self.known_face_names = pickle.load(file)
        print(f"Encodings loaded from {filename}")

    def calculate_ranges_for_all_classes(self):
        """Calculate the acceptable distance range for each class."""
        self.class_ranges = {}
        for class_name in set(self.known_face_names):
            # Collect all distances for the class
            distances = []
            for i, name in enumerate(self.known_face_names):
                if name == class_name:
                    for encoding in self.known_face_encodings:
                        distance = face_recognition.face_distance([encoding], self.known_face_encodings[i])[0]
                        distances.append(distance)
            if distances:
                self.class_ranges[class_name] = (min(distances), max(distances))
                print(f"Range for {class_name}: {self.class_ranges[class_name]}")

    def is_in_person_range(self, person_name, distance):
        """Check if the distance is within the acceptable range for a person."""
        if person_name in self.class_ranges:
            min_range, max_range = self.class_ranges[person_name]
            return min_range <= distance <= max_range
        return False

    def detect_known_faces(self, frame):
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find face locations and encodings in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Calculate distances to all known faces
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            # Find the best match index
            best_match_index = np.argmin(face_distances)
            best_match_name = self.known_face_names[best_match_index]
            best_match_distance = face_distances[best_match_index]

            # Check if the best match is within the acceptable range
            if self.is_in_person_range(best_match_name, best_match_distance):
                face_names.append(best_match_name)  # Only include matches within range
            else:
                face_names.append("Unknown")  # Mark as unknown if out of range

        # Convert face locations to the original frame size
        face_locations = np.array(face_locations) / self.frame_resizing
        return face_locations.astype(int), face_names


# sfr = SimpleFacerec()
# sfr.load_encoding_images("Database")  # Specify your folder path here
# sfr.save_encodings("face_encodings.pkl")  # Save encodings to a file

# # Example Usage
# sfr = SimpleFacerec()
# sfr.load_saved_encodings("face_encodings.pkl")  # Load saved encodings

# # Automatically calculate ranges for all classes (persons)
# sfr.calculate_ranges_for_all_classes()

# # Detect faces in a test frame
# frame = cv2.imread("DataBase/mohamed H/mohamed henedy.jpg")  # Replace with your live frame capture
# face_locations, face_names = sfr.detect_known_faces(frame)

# # Print the results
# print(face_names)
