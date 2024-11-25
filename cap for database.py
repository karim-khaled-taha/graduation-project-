import cv2 as cv

# Define the camera capture and parameters
camera = cv.VideoCapture(0)
camera.set(3, 1280)  # Set the width of the webcam frame
camera.set(4, 720)   # Set the height of the webcam frame

fonts = cv.FONT_HERSHEY_COMPLEX
GREEN = (0, 255, 0)

capture = False
number = 0

while True:
    ret, frame = camera.read()
    frame = cv.flip(frame, 1)
    if not ret:
        print("Failed to grab frame")
        break
    cv.putText(frame, "Press Enter to Capture", (50, 100), fonts, 1, GREEN, 2)

    # Display feedback if capture is triggered
    if capture:
        cv.putText(frame, "Image Captured!", (50, 50), fonts, 1, GREEN, 2)
        capture = False  # Reset capture flag

    # Display the video feed
    cv.imshow('frame', frame)

    key = cv.waitKey(1)

    # Capture image when 'Enter' is pressed
    if key == 13:  # Enter key
        number += 1
        image_path = f'ReferenceImages/image{number}.png'
        cv.imwrite(image_path, frame)
        print(f'Image saved as {image_path}')
        capture = True

    # Exit loop when 'q' is pressed
    if key == ord('q'):
        break

# Release camera and close windows
camera.release()
cv.destroyAllWindows()
