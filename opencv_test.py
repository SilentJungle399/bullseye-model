import cv2
import os
import numpy as np

def detect_bullseye(image_path):
    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Image not found!")
        return

    # Resize for consistency
    frame = cv2.resize(frame, (640, 640))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use morphological operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

    # Apply Canny edge detection
    edges = cv2.Canny(cleaned, 50, 150)

    # Detect circles using Hough Transform
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=0.001,
        param1=300, param2=35, minRadius=20, maxRadius=150
    )
    x = 1
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw detected circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Put the order number at the center
            cv2.putText(frame, str(x), (i[0] - 10, i[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            print(f"Detected Bullseye at: {i[0]}, {i[1]} (Radius: {i[2]})")
            x += 1

            if x % 5 == 0:
                cv2.imshow("Bullseye Detection", frame)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if x > 20:
                break

# Test with multiple images
for img in os.listdir("examples"):
    detect_bullseye(f"examples/{img}")
