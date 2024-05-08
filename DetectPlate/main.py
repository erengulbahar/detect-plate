import cv2
import os

# Frame dimensions
frameWidth = 500
frameHeight = 480
brightness = 150

CASCADE_PATH = "haarcascade_russian_plate_number.xml"

# Load the plate classifier
plateCascade = cv2.CascadeClassifier(CASCADE_PATH)
minArea = 500

# Video capture
cap = cv2.VideoCapture("./videos/video.mp4")

# Set camera properties
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

count = 0
save_key = ord('s')
wait_duration = 500

while True:
    success, img = cap.read()
    if not success:
        break  # Stop the loop if there's a failure or no more frames

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

            imgRoi = img[y:y+h, x:x+w]
            cv2.imshow("Number Plate", imgRoi)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == save_key:
        img_path = os.path.join("images", f"{count}.jpg")
        cv2.imwrite(img_path, imgRoi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(wait_duration)
        count += 1

cv2.destroyAllWindows()
cap.release()