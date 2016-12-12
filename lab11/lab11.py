import cv2
import numpy as np
from matplotlib import pyplot as plt
from drawMatches import drawMatches

# Change the following variables to any other two image names
# For example Pyramids1.jpg and Pyramids2.jpg
imgName1 = 'GMIT1.jpg'
imgName2 = 'GMIT2.jpg'

# Using OpenCV, convert the GMIT1 image to gray scale
imgOrig1 = cv2.imread(imgName1)
imgGray1 = cv2.cvtColor(imgOrig1, cv2.COLOR_BGR2GRAY)

# Plot the output image to verify that it is indeed grayscale
plt.subplot(2, 1, 1)
plt.imshow(cv2.cvtColor(imgOrig1, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 1, 2)
plt.imshow(imgGray1, cmap = 'gray')
plt.title('GrayScale')
plt.xticks([])
plt.yticks([])

plt.show()

# Perform Harris corner detection on the grayscale input image
imgHarris = imgOrig1.copy()
pst = cv2.cornerHarris(imgGray1, 2, 3, 0.04)

# Loop through every element in the 2d matrix pst1.
# If the element is greater than a threshold, draw a circle on the image.
threshold = 0.20 # number between 0 and 1

for i in range(len(pst)):
	for j in range(len(pst[i])):
		if pst[i][j] > (threshold * pst.max()):
			cv2.circle(imgHarris, (j,i), 3, (0, 255, 255), -1)

# Perform corner detection using the Shi Tomasi algorithm
imgShiTomasi = imgOrig1.copy()
corners = cv2.goodFeaturesToTrack(imgGray1, 50, 0.01, 10)

# Loop through the corners array and plot a circle at each corner
for i in corners:
	x,y = i.ravel()
	cv2.circle(imgShiTomasi, (x,y), 3, (255, 0, 0), -1)

# Initiate ORB-SIFT detector and only return 50 features at a maximum (rather than 500 which is the default)
orb = cv2.ORB(50)
# Find the keypoints and descriptors with ORB-SIFT
kp1, des1 = orb.detectAndCompute(imgGray1, None)
# Draw only keypoints location, not size and orientation
imgOrb = cv2.drawKeypoints(imgOrig1, kp1, color=(0, 0, 255))

# Plot imgHarris
plt.subplot(3, 1, 1)
plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Harris')
plt.xticks([])
plt.yticks([])		

# Plot imgShiTomasi
plt.subplot(3, 1, 2)
plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Shi Tomasi')
plt.xticks([])
plt.yticks([])

# Plot imgOrb
plt.subplot(3, 1, 3)
plt.imshow(cv2.cvtColor(imgOrb, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Orb')
plt.xticks([])
plt.yticks([])

plt.show()

# Demonstrate feature matching
# Convert the second image to grayscale
imgOrig2 = cv2.imread(imgName2)
imgGray2 = cv2.cvtColor(imgOrig2, cv2.COLOR_BGR2GRAY)

kp2, des2 = orb.detectAndCompute(imgGray2, None)
# Draw only keypoints location, not size and orientation
imgOrb2 = cv2.drawKeypoints(imgOrig2, kp1, color=(0, 0, 255))

# Brute force matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort them in the order of their distance
matches = sorted(matches, key = lambda x:x.distance)

img3 = drawMatches(imgGray1, kp1, imgGray2, kp2, matches[:20])