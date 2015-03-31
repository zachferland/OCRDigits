import cv2
import numpy as np

# Given a string of the path of an image, this returns an array of images, each of which 
# are single characters formatted like images in the MNIST data set. Order left to righ.
# This currently only reads a single line of numbers in an image, could easily be
# extended to read multiple

def getNumbers(image):
	img = cv2.imread(image)

	# Extend this to handle determining if a img has to be rotated or not
	# take horizontal and vertical histogram, rotate so that horizontal has greater distribution
	img = cv2.transpose(img);
	img = cv2.flip(img, 1);

	processed = _preProcess(img)
	segments = _charSegmentation(processed, img)
	numbers = _normalizeFormat(segments)

	return numbers

# Given an instance of an image(numpy array), process the image to make 
# image segmentation easier and more accurate.
def _preProcess(image):	
	# Convert to Gray Scale
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	
	# Remove noise (maybe too much if numbers written smaller?)
	gray = cv2.medianBlur(gray,11)

	# Deskewing can be added for improved accuracy
	
	# Apply adaptive threshold
	thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
	
	# Smooth
	smoothed = cv2.dilate(thresh,None,iterations = 3)
	smoothed = cv2.erode(smoothed,None,iterations = 2)
	processed = cv2.medianBlur(smoothed,11)

	return processed

# Give a Countour returns x coordinate of lower left corner
def _xsort(cnt):
	x,y,w,h = cv2.boundingRect(cnt)
	return x

# Given a appropiately pre-processed image and the original, this will return an
# array of images, each image is a single character, sorted along x axis

def _charSegmentation(img, original):
	# Character Segmentations with countours
	contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	# Sort countours from left to right
	contours.sort(key = lambda cnt: _xsort(cnt))
	
	# Seperate each countour into its own image
	numbers = []
	
	for cnt in contours:
	    x,y,w,h = cv2.boundingRect(cnt)
	    crop = original[y:y+h, x:x+w]
	    crop = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
	    # May not need blurring and may depend on size of numbers in text
	    crop = cv2.medianBlur(crop, 11) 
	    ret,crop = cv2.threshold(crop,99,255,cv2.THRESH_BINARY)
	    crop = cv2.medianBlur(crop, 11)
	
	    numbers.append(crop)

	return numbers

# Given an array of images, there sizes and format are normalized. Specifically
# in this instance the images are formatted to meet the specs in the MNIST data set.
# Each character is greyscaled and resized with ani-aliasing to fit in 20*20. Then the
# the mass center of the image is found and used to center it in a 28*28 image.

def _normalizeFormat(numbers):
	finalImages = []

	for number in numbers:
		# Resize each image to fit in 20*20
		yratio = float(20) / float(number.shape[0])
		xratio = float(20) / float(number.shape[1])
		ratio = min(yratio, xratio)

		# Resize
		resized = cv2.resize(number, (0,0), fx=ratio, fy=ratio) 

		# Thick/Smooth
		smoothed = cv2.GaussianBlur(resized,(3,3),0)

		# Invert Image
		inverted = abs(255 - smoothed)

		# Increase Contrast
		for r in xrange(inverted.shape[0]):
			for c in xrange(inverted.shape[1]):
				inverted[r][c] = min(round(inverted[r][c] * 1.9), 255)

		#Determine Mass Center of Image
		small = inverted
		moments = cv2.moments(small)
		x = int(round(moments['m10'] / moments['m00']))
		y = int(round(moments['m01'] / moments['m00']))
		massCenter = (x,y)
		
		#Using mass center place image at center of 28*28 image
		finalImage = np.zeros((28, 28), np.uint8)

		# Determine sub grid in 28 * 28
		y1 = max(14-massCenter[1], 0)
		y2 = min(14+(small.shape[0]-massCenter[1]), 28)
		x1 = max(14-massCenter[0], 0)
		x2 = min(14+(small.shape[1]-massCenter[0]), 28)

		# Gurantee it matches dimensions of small image
		while not (y2 - y1) == small.shape[0]:
			if not y2 == 28:
				y2 += 1
			else:
				y1 -= 1

		while not (x2 - x1) == small.shape[1]: 
			if not x2 == 28:
				x2 += 1
			else:
				x1 -= 1

		finalImage[y1:y2, x1:x2] = small

		finalImages.append(finalImage)

	return finalImages