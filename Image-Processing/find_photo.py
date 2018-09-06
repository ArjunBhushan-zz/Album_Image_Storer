# -*- coding: utf-8 -*-

import numpy as np
import cv2 as cv
import os

def greyscale_images():
    for file in os.listdir('./test_images'):
        grey_scale_img = cv.imread('./test_images/' + file, 
                                   cv.IMREAD_GRAYSCALE)
        cv.imwrite('./transformed_images/grey_scale/' + file, 
                   grey_scale_img)

def canny_images():
    for file in os.listdir('./transformed_images/grey_scale'):
        grey_scale = cv.imread('./transformed_images/grey_scale/' + file,
                               cv.IMREAD_GRAYSCALE)
        grey_scale_without_noise = cv.GaussianBlur(grey_scale, (15,15), 0)
        th, clear_grey_img = cv.threshold (grey_scale_without_noise, 0, 255,
                                           cv.THRESH_BINARY+cv.THRESH_OTSU)
        canny_img = cv.Canny(clear_grey_img, th/3, th)
        cv.imwrite('./transformed_images/canny/' + file, canny_img)

def contour_images():
    for file in os.listdir('./transformed_images/canny'):
        canny_img = cv.imread('./transformed_images/canny/' + file,
                              cv.IMREAD_GRAYSCALE)
        canny_copy = canny_img.copy()
        _, contours, _ = cv.findContours(canny_img, cv.RETR_TREE , 
                                                   cv.CHAIN_APPROX_NONE)
    
        #epsilon = 0.8*cv.arcLength(contours[0], True)
        #approx = cv.approxPolyDP(contours[0], epsilon, True)
        if (len(contours)):
            max_area = cv.contourArea(contours[0])
            max_contour = cv.contourArea(contours[0])
            for contour in contours:
                if (cv.contourArea(contour) > max_area):
                    max_contour = contour
            print(max_contour)
            
        cv.drawContours(canny_copy, contours, -1, (128,0,0), 12)
        cv.imwrite('./transformed_images/contour/' + file, canny_copy)
    
if (__name__ == '__main__'):
    # get the greyscale of the images and store it in ./transformed_images/grey_scale/
    greyscale_images()
    # Find edges:
    # remove noise - Gaussian, threshold - Otsu, and find lines with Canny Edge Detection
    #canny_images()
    # only keep distinct connected shapes
    contour_images()