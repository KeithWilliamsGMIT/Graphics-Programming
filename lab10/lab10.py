import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# Import a colour image (Also tried the Big Ben image)
imgOrig = cv2.imread('GMIT.jpg')

# Convert to gray scale
imgGray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)

# Plot the original and gray images
plt.subplot(2, 1, 1)
plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 1, 2)
plt.imshow(imgGray, cmap = 'gray')
plt.title('GrayScale')
plt.xticks([])
plt.yticks([])

plt.show()

# Blur
imgBlurSmall = cv2.GaussianBlur(imgGray,(5, 5),0)
imgBlurLarge = cv2.GaussianBlur(imgGray,(15, 15),0)

# Plot the original, gray and blurred images
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 2)
plt.imshow(imgGray, cmap = 'gray')
plt.title('GrayScale')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 3)
plt.imshow(imgBlurSmall, cmap = 'gray')
plt.title('Blur5')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 2, 4)
plt.imshow(imgBlurLarge, cmap = 'gray')
plt.title('Blur10')
plt.xticks([])
plt.yticks([])

plt.show()

# Edge Detection
sobelVertical = cv2.Sobel(imgGray, cv2.CV_64F, 1, 0, ksize=5)	# x dir
sobelHorizontal = cv2.Sobel(imgGray, cv2.CV_64F, 0, 1, ksize=5)	# y dir
sobelSum = sobelHorizontal + sobelVertical
canny = cv2.Canny(imgGray, 40, 200)

# Plot sobel images
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 2)
plt.imshow(imgGray, cmap = 'gray')
plt.title('GrayScale')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 3)
plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title('Sobel Horizontal')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 4)
plt.imshow(sobelVertical, cmap = 'gray')
plt.title('Sobel Vertical')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 5)
plt.imshow(sobelSum, cmap = 'gray')
plt.title('Sobel Sum')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 3, 6)
plt.imshow(canny, cmap = 'gray')
plt.title('Canny')
plt.xticks([])
plt.yticks([])

plt.show()

# Advanced exercises
# Select a threshold for the Sobel sum image and set all the values below the threshold to 0 and all above the threshold to 255
height, width = sobelSum.shape
treshold = 20

for i in range(0, height):
    for j in range(0, width):
		# Clamp the pixel value between 0 and 255
		# Step 1) Get the absolute value of the pixel
		# Step 2) Divide by depth of the image (64)
		# Step 3) Cast the pixel value to an int
		sobelSum[i,j] = int(abs(sobelSum[i,j]) / 64)
		
		if abs(sobelSum[i,j]) < treshold:
			sobelSum[i,j] = 0
		else:
			sobelSum[i,j] = 255

# Plot results sombel sum with treshold and own edge detection
plt.subplot(1, 1, 1)
plt.imshow(sobelSum, cmap = 'gray')
plt.title('Sobel Sum With Treshold')
plt.xticks([])
plt.yticks([])

plt.show()

# Manually write your own edge detector using a first derivative
img = imgGray.copy()
width, height = img.shape

# The following nested for loop in an edge detection algorithm
# It iterates over every pixel in the gray scale image
# It calculates gx and gy my multiplying the neighbouring pixels by a 3x3 matrix
# gx => [-1, 0, 1] gy => [ 1, 2, 1]
# 		[-2, 0, 2]		 [ 0, 0, 0]
# 		[-1, 0, 1]		 [-1,-2,-1]
# The values are accumulated
# The length of the gradient calculated using pythagoras' theorem
# If the length exceeds the treshold it's on an egde
imgEdges = img.copy()
treshold = 200

for x in range(1, width - 1):
	for y in range(1, height - 1):
		# initialise Gx to 0 and Gy to 0 for every pixel
		gx = 0
		gy = 0
		
		# accumulate the value of the top left pixel (0 - 255) into gx, and yy
		gx += -1 * img[x - 1, y - 1]
		gy += -1 * img[x - 1, y - 1]

		# now we do the same for the remaining pixels, left to right, top to bottom

		# remaining left column
		gx += -2 * img[x - 1, y]
		gx += -1 * img[x - 1, y + 1]
		gy += img[x - 1, y + 1]

		# middle pixels
		gy += -2 * img[x, y - 1]
		gy += 2 * img[x, y + 1]

		# right column
		gx += img[x + 1, y - 1]
		gy += -1 * img[x + 1, y - 1]
		gx += 2 * img[x + 1, y]
		gx += img[x + 1, y + 1]
		gy += img[x + 1, y + 1]
		
		# calculate the length of the gradient
		length = math.sqrt((gx * gx) + (gy * gy))
		
		if length > treshold:
			length = 255
		else:
			length = 0
		
		imgEdges[x, y] = int(length)

plt.subplot(2, 1, 1)
plt.imshow(img, cmap = 'gray')
plt.title('GrayScale')
plt.xticks([])
plt.yticks([])

plt.subplot(2, 1, 2)
plt.imshow(imgEdges, cmap = 'gray')
plt.title('Manual Edge Detection With Treshold')
plt.xticks([])
plt.yticks([])

plt.show()