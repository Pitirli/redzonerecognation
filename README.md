from ultralytics import YOLO  # Import YOLO model
import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
from PIL import Image  # Pillow for image handling

# Load the YOLO model by specifying the model path
model = YOLO("C:/Users/YUSUF ALPTUG PITIRLI/PycharmProjects/YOLOv8/best (1).pt")

# Start the camera capture (index 0 refers to the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Camera could not be opened")
    exit()

# Infinite loop to continuously read frames from the webcam
while True:
    ret, frame = cap.read()  # Read a frame from the camera
    if not ret:  # If no frame is captured, exit the loop
        print("No frame detected")
        break

    # Define red color in BGR format
    red = [0, 0, 255]

    # Convert the captured frame from BGR to HSV color space
    hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper limits for the red color in HSV (first range)
    lowerLimit = ([0, 160, 160])  # Lower bound for red color
    upperLimit = ([10, 255, 255])  # Upper bound for red color
    lowerLimit = np.array(lowerLimit, dtype=np.uint8)  # Convert to NumPy array
    upperLimit = np.array(upperLimit, dtype=np.uint8)  # Convert to NumPy array

    # Create a mask for pixels within the red color range (first range)
    mask0 = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Define the lower and upper limits for the red color in HSV (second range)
    lowerLimit = ([175, 160, 160])  # Lower bound for red color
    upperLimit = ([180, 255, 255])  # Upper bound for red color
    lowerLimit = np.array(lowerLimit, dtype=np.uint8)  # Convert to NumPy array
    upperLimit = np.array(upperLimit, dtype=np.uint8)  # Convert to NumPy array

    # Create a mask for pixels within the red color range (second range)
    mask1 = cv2.inRange(hsvImage, lowerLimit, upperLimit)

    # Combine the two masks to detect the entire red color range
    mask = mask0 + mask1

    # Convert the mask to a Pillow Image object
    mask_ = Image.fromarray(mask)

    # Get the bounding box of the detected red region
    bbox = mask_.getbbox()

    # If a bounding box is detected, draw a red rectangle around the region
    if bbox is not None:
        x1, y1, x2, y2 = bbox  # Get the coordinates of the bounding box
        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Draw the rectangle

    # Apply the YOLO model to the frame to detect objects
    result = model(frame)

    # Plot the detected objects on the frame
    drawn_frame = result[0].plot()

    # Display the frame with the detected objects
    cv2.imshow("Live Detection", drawn_frame)

    # Break the loop if the ESC key is pressed (ASCII value 27)
    if cv2.waitKey(1) == 27:
        break

# Release the camera resource
cap.release()

# Close the OpenCV window
cv2.destroyAllWindows()
