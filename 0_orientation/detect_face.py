import cv2

# load image and convert to gray scale
image = cv2.imread("IMG_2294.JPG")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# load the face detector and detect face
# Note: Make sure to have the correct file content + correct path!
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rects = detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, flags=cv2.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


# show image
cv2.imshow("Faces", image)
cv2.waitKey(0)