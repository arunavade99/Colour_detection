import cv2
import numpy as np

# Define HSV color range (your working one)
lower_hsv = np.array([132, 45, 118])
upper_hsv = np.array([179, 193, 255])
kernel = np.ones((5, 5), np.uint8)

cap = cv2.VideoCapture("(1).mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Apply your filtering steps
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detection = False
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 300:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Metal", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (102, 0, 255), 2)
            detection = True

    print("True" if detection else "False")

    # Resize for display
    frame_resized = cv2.resize(frame, (500, 500))
    mask_resized = cv2.resize(mask, (500, 500))

    cv2.imshow('Molten Metal Detection', frame_resized)
    cv2.imshow('Mask', mask_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
