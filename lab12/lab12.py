import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load an image into OpenCV and convert it to gray scale
imgOrig = cv2.imread('faces.jpg')
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Open classifiers (Downloaded from https://github.com/opencv/opencv/tree/master/data/haarcascades)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
frontalcatface_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')

# Face detection (Adapted from http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#facedetection)
faces = face_cascade.detectMultiScale(imgGray, 1.1, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(imgOrig,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = imgGray[y:y+h, x:x+w]
    roi_color = imgOrig[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow('img',imgOrig)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load the image into OpenCV
imgOrig = cv2.imread('cat.jpg')
imgCopy = imgOrig.copy()

# Convert it to gray scale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Upper body detection
catfaces = frontalcatface_cascade.detectMultiScale(imgGray, 1.1, 5)
for (x,y,w,h) in catfaces:
	# Draw on the copy so that the original can be used later
    cv2.rectangle(imgCopy,(x,y),(x+w,y+h),(0,0,255),2)

# Show the copy
cv2.imshow('img',imgCopy)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Split the image into R,G and B channels
imgRed = imgOrig.copy()
imgGreen = imgOrig.copy()
imgBlue = imgOrig.copy()

for i in range(len(imgOrig)):
	for j in range(len(imgOrig[i])):
		for k in range(len(imgOrig[i][j])):
			if k == 0:
				# Set blue channel for green and red images to 0
				imgGreen[i][j][k] = 0
				imgRed[i][j][k] = 0
			elif k == 1:
				# Set green channel for blue and red images to 0
				imgBlue[i][j][k] = 0
				imgRed[i][j][k] = 0
			else:
				# Set red channel for blue and green images to 0
				imgBlue[i][j][k] = 0
				imgGreen[i][j][k] = 0

# View the result using a matplotlib subplot 
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(imgRed, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Red')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(imgGreen, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Green')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(imgBlue, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Blue')
plt.xticks([])
plt.yticks([])

plt.show()

# Transform the image into the HSV colour space
imgHsv = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2HSV)

# Split the image into the separate HSV channels
h = imgHsv.copy()
s = imgHsv.copy()
v = imgHsv.copy()

for i in range(len(imgOrig)):
	for j in range(len(imgOrig[i])):
		for k in range(len(imgOrig[i][j])):
			if k == 0:
				# Set the hue in s and v
				s[i][j][k] = 0
				v[i][j][k] = 0
			elif k == 1:
				# Set the saturation in h and v
				h[i][j][k] = 255
				v[i][j][k] = 0
			else:
				# Set the value in h and s
				h[i][j][k] = 255
				s[i][j][k] = 255

# View the resulting images
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(imgHsv, cv2.COLOR_HSV2RGB), cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(h, cv2.COLOR_HSV2RGB), cmap = 'gray')
plt.title('Hue')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(s, cv2.COLOR_HSV2RGB), cmap = 'gray')
plt.title('Saturation')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(v, cv2.COLOR_HSV2RGB), cmap = 'gray')
plt.title('Value')
plt.xticks([])
plt.yticks([])

plt.show()

# Perform the same transformation and channel separation for the HLS space
# Transform the image into the HLS colour space
imgHls = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2HLS)

# Split the image into the separate HLS channels
h = imgHls.copy()
l = imgHls.copy()
s = imgHls.copy()

for i in range(len(imgOrig)):
	for j in range(len(imgOrig[i])):
		for k in range(len(imgOrig[i][j])):
			if k == 0:
				# Set the hue in l and s
				s[i][j][k] = 0
				l[i][j][k] = 0
			elif k == 1:
				# Set the lightness in h and s
				h[i][j][k] = 128
				s[i][j][k] = 128
			else:
				# Set the saturation in h and l
				h[i][j][k] = 255
				l[i][j][k] = 0

# View the resulting images
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(imgHls, cv2.COLOR_HLS2RGB), cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(h, cv2.COLOR_HLS2RGB), cmap = 'gray')
plt.title('Hue')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(l, cv2.COLOR_HLS2RGB), cmap = 'gray')
plt.title('Lightness')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(s, cv2.COLOR_HLS2RGB), cmap = 'gray')
plt.title('Saturation')
plt.xticks([])
plt.yticks([])

plt.show()