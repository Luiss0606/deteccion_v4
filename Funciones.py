import cv2
import numpy as np

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# resize the image
def get_resize(image):
    return cv2.resize(image, None, fx=7, fy=7, interpolation=cv2.INTER_NEAREST_EXACT)

# noise removal
def remove_noise(image):
    return cv2.bilateralFilter(image, 7, 45, 45)
    #return cv2.bilateralFilter(image, 5, 35, 35)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]


# dilation
def get_dilate(image):
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    return cv2.dilate(image, rect_kern, iterations=1)


# erosion
def get_erode(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.erode(image, kernel, iterations=1)

# Opening image
def get_open(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)) #Se usa rect
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# Closing image
def get_close(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)



def most_freq(list):
    if len(list) >= 1:
        num0 = max(set(list), key = list.count)
        return num0

def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if 47<ord(c)<58 or 64<ord(c)<91 else "" for c in text]).strip()



