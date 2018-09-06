import numpy as np
import cv2 as cv
import os

def remove_border(orig_img):
    # cited: http://artsy.github.io/blog/2014/09/24/using-pattern-recognition
    #-to-automatically-crop-framed-art/ for optimized method
    cv.imwrite('./transformed_images/test.jpg', orig_img)
    
    # Gray out image
    
    gray_img = cv.cvtColor(orig_img, cv.COLOR_BGR2GRAY)
    
    # Dilation
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (5,5))
    eroded_img = cv.dilate(gray_img ,kernel,iterations = 1)
    cv.imwrite('./transformed_images/test.jpg', eroded_img)
    
    # Media Blurring
    blurred_img = cv.medianBlur(eroded_img,5)
    cv.imwrite('./transformed_images/test.jpg', blurred_img)
    
    # Shrink
    #shrinked_img = cv.resize(blurred_img, None, fx=0.5, fy=0.5, \
     #                        interpolation = cv.INTER_AREA)
    #cv.imwrite('./transformed_images/test.jpg', shrinked_img)
    # Zoom
    #zoomed_img = cv.resize(blurred_img, None, fx=2, fy=2, \
     #                      interpolation = cv.INTER_CUBIC)
    #cv.imwrite('./transformed_images/test.jpg', zoomed_img)
    
    # Canny edge detection
    th = 100
    canny_img = cv.Canny(blurred_img, th/3, th)
    cv.imwrite('./transformed_images/test.jpg', canny_img)
    
    # Dilate the image again to thicken the edges
    edge_img = cv.dilate(canny_img, kernel,iterations = 1)
    cv.imwrite('./transformed_images/test.jpg', edge_img)
    
    # Find all contours
    _, contours, _ = cv.findContours(canny_img.copy(), cv.RETR_TREE, 
                                                   cv.CHAIN_APPROX_SIMPLE)
    # Remove contours that are not a rectangle
    #for contour_index in reversed(range(len(contours))):
     #   if (not(len(contours[contour_index]) == 4)):
      #      contours = np.delete(contours, contour_index)
    # Draw all remaining contours
    #cv.drawContours(orig_img, contours, -1, (0,255,0), 6)
    
    # get largest contour and draw its approximation
    largest_area = cv.contourArea(contours[0])
    largest_contour = 0
    for contour_index in range(len(contours)):
        if (cv.contourArea(contours[contour_index]) > largest_area):
            largest_contour = contour_index
    cnt = contours[largest_contour]
    epsilon = 0.1*cv.arcLength(cnt,True)
    approx = cv.approxPolyDP(cnt,epsilon,True)
    cv.drawContours(orig_img, [approx], -1, (0,255,0), 6)
    cv.imwrite('./transformed_images/test.jpg',orig_img)
    
if (__name__ == '__main__'):
    # Send in the greyscaled image
    img = remove_border(cv.imread('./test_images/IMG_20180905_115014.jpg'))