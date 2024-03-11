import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import easyocr

# Load image
image = cv2.imread("image2.jpg")

# Convert image to grayscale
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show image
plt.imshow(cv2.cvtColor(grayscale_image, cv2.COLOR_BGR2RGB))

plt.show()

# Filters
bfilter = cv2.bilateralFilter(grayscale_image, 11, 11, 17)
edge = cv2.Canny(bfilter, 30, 200)

# Black & White image
plt.imshow(cv2.cvtColor(edge, cv2.COLOR_BGR2RGB))

plt.show()

# Contours
keypoints = cv2.findContours(edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]

location = None

for contour in contours:
  # cv2.approxPolyDP returns a resampled contour, so this will still return a set of (x, y) points
  approx = cv2.approxPolyDP(contour, 10, True)
  if len(approx) == 4:
    location = approx
    break

# Masking
mask = np.zeros(grayscale_image.shape, np.uint8)
new_image = cv2.drawContours(mask, [location], 0, 255, -1)
new_image = cv2.bitwise_and(image, image, mask = mask)

# Show number plate image
plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))

plt.show()

(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))

# Adding buffer
crop_image = grayscale_image[x1:x2+3, y1:y2+3]

# Cropped number plate image
plt.imshow(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))

plt.show()

# Get number plate image to text
reader = easyocr.Reader(["en"])
result = reader.readtext(crop_image)
text = result[0][1]
font = cv2.FONT_HERSHEY_SIMPLEX

res = cv2.putText(image, text = text, org = (approx[0][0][0], approx[1][0][1]+60), fontFace = font, fontScale = 1, color = (0, 255, 0), thickness = 5)
res = cv2.rectangle(image, tuple(approx[0][0]), tuple(approx[2][0]), (0, 255, 0), 3)

# Put text to below number plate
plt.imshow(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))

plt.show()