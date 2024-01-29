import cv2
import os
import time
import pytesseract
import re
import numpy as np

# Load the Haar Cascade classifier for plate detection
harcascade = "model/haarcascade_russian_plate_number.xml"
plate_cascade = cv2.CascadeClassifier(harcascade)

# Tesseract executable path (update this path according to your installation)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Username and password for login
username = "admin"
password = "admin"  # You should use a more secure password in a real-world scenario

# Prompt the user for login
input_username = input("Enter username: ")
input_password = input("Enter password: ")

if input_username != username or input_password != password:
    print("Invalid username or password. Exiting.")
    exit(1)

# Open the video capture stream (default camera)
try:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Failed to open camera. Check camera availability and index.")
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Set the capture frame dimensions
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

min_area = 500
save_folder = "plates"

# Create a folder to save the detected plates if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Variable to check if an image has already been saved
image_saved = False

while True:
    # Capture a frame from the camera
    ret, img = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect license plates
    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area and not image_saved:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)

            img_roi = img[y: y + h, x: x + w]

            # Preprocess the image (resize, denoise, threshold, etc.)
            img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

            # Adaptive thresholding
            _, img_thresh = cv2.threshold(img_roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours based on area, and get the largest contour
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            largest_contour = contours[0]

            # Get the bounding box of the largest contour
            x_, y_, w_, h_ = cv2.boundingRect(largest_contour)

            # Ensure the license plate area covers a significant portion of the bounding box
            plate_area_ratio = (w_ * h_) / (w * h)
            if plate_area_ratio > 0.7:
                # Crop the license plate region
                license_plate_roi = img_roi[y_:y_ + h_, x_:x_ + w_]

                # Convert the license plate image to grayscale
                gray_license_plate = cv2.cvtColor(license_plate_roi, cv2.COLOR_BGR2GRAY)

                # Find edges in the license plate image
                edges = cv2.Canny(gray_license_plate, 50, 150)

                # Display the license plate edges
                cv2.imshow('License Plate Edges', edges)

                # Use Tesseract OCR to recognize text on the license plate
                plate_text = pytesseract.image_to_string(gray_license_plate, config='--psm 11')

                # Filter and clean the recognized text
                plate_text = re.sub(r'[^A-Z0-9]', '', plate_text)

                if plate_text.strip():
                    print(f"License Plate Text: {plate_text.strip()}")

                    # Save the image only if the plate text is present
                    timestamp = int(time.time())
                    filename = os.path.join(save_folder, f"plate_{timestamp}.jpg")
                    cv2.imwrite(filename, license_plate_roi)
                    print(f"Image saved: {filename}")

                    # Set the flag to indicate that the image has been saved
                    image_saved = True

    # Display the captured frame in a window
    cv2.imshow('Camera Feed', img)

    key = cv2.waitKey(1)

    if key & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
