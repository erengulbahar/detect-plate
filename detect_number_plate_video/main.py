import cv2

# Import haarcascade model
plat_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Load video
video = cv2.VideoCapture('./videos/video.mp4')

# Check video is open or close
if (video.isOpened() == False):
    print('Error Reading Video')

# Run until video finishing
while True:
    # Frame is capture every seconds, ret status
    ret, frame = video.read()

    # Convert to grayscale video from RGB
    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect plate
    plate = plat_detector.detectMultiScale(gray_video, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25))

    for (x, y, w, h) in plate:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        frame[y:y + h, x:x + w] = cv2.blur(frame[y:y + h, x:x + w], ksize=(10, 10))

        # Put the text to number plate
        cv2.putText(frame, text='License Plate', org=(x - 3, y - 3), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    color=(0, 0, 255), thickness=1, fontScale=0.6)

    if ret == True:
        cv2.imshow('Video', frame)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()