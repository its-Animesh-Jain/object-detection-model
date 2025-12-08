import cv2
import numpy as np
import os

def create_better_test_image():
    """Create a test image with objects that YOLO can recognize better"""
    # Create a more realistic background
    image = np.ones((600, 800, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Draw objects that resemble common COCO dataset objects
    
    # 1. Person-like silhouette (stick figure)
    cv2.rectangle(image, (100, 300), (120, 500), (0, 0, 0), -1)  # Body
    cv2.circle(image, (110, 280), 20, (0, 0, 0), -1)  # Head
    cv2.line(image, (110, 350), (80, 400), (0, 0, 0), 3)  # Left arm
    cv2.line(image, (110, 350), (140, 400), (0, 0, 0), 3)  # Right arm
    cv2.line(image, (110, 500), (90, 550), (0, 0, 0), 3)  # Left leg
    cv2.line(image, (110, 500), (130, 550), (0, 0, 0), 3)  # Right leg
    
    # 2. Car-like shape
    car_body = np.array([[400, 300], [600, 300], [620, 350], [380, 350]], np.int32)
    cv2.fillPoly(image, [car_body], (255, 0, 0))  # Blue car
    cv2.rectangle(image, (420, 250), (480, 300), (200, 200, 200), -1)  # Window
    cv2.rectangle(image, (520, 250), (580, 300), (200, 200, 200), -1)  # Window
    cv2.circle(image, (450, 370), 15, (0, 0, 0), -1)  # Wheel
    cv2.circle(image, (550, 370), 15, (0, 0, 0), -1)  # Wheel
    
    # 3. Chair-like shape
    cv2.rectangle(image, (200, 400), (280, 450), (0, 100, 0), -1)  # Seat
    cv2.rectangle(image, (200, 350), (220, 400), (0, 150, 0), -1)  # Back left
    cv2.rectangle(image, (260, 350), (280, 400), (0, 150, 0), -1)  # Back right
    cv2.rectangle(image, (210, 450), (230, 500), (0, 120, 0), -1)  # Leg
    cv2.rectangle(image, (250, 450), (270, 500), (0, 120, 0), -1)  # Leg
    
    # 4. Book/rectangle object
    cv2.rectangle(image, (650, 200), (750, 300), (0, 0, 255), -1)  # Red book
    cv2.line(image, (650, 200), (750, 200), (255, 255, 255), 2)  # Top edge
    cv2.line(image, (750, 200), (750, 300), (255, 255, 255), 2)  # Right edge
    
    # 5. Bottle-like shape
    cv2.rectangle(image, (300, 200), (320, 350), (255, 255, 0), -1)  # Bottle body
    cv2.rectangle(image, (290, 190), (330, 200), (200, 200, 100), -1) # Bottle neck
    cv2.rectangle(image, (295, 180), (325, 190), (150, 150, 80), -1)  # Bottle cap
    
    # Add labels for reference
    cv2.putText(image, "Person-like", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, "Car-like", (450, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "Chair-like", (180, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, "Book-like", (650, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, "Bottle-like", (280, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.putText(image, "Better Test Image for YOLO Detection", (200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.imwrite('better_test_image.jpg', image)
    print("✅ Created 'better_test_image.jpg' with more recognizable shapes")
    print("📐 Contains: Person-like, Car-like, Chair-like, Book-like, Bottle-like shapes")
    
    return image

if __name__ == "__main__":
    create_better_test_image()