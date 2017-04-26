#!/usr/bin/env python
import roslib; roslib.load_manifest('cv_test')
import rospy
import pickle
import numpy as np
import cv2
import sys
import glob, os
import math

x_half = 320
y_half = 240

def make_input(data):
    assert data.shape[1] == 3
    # make data into 0's/1's
    data = data.astype(np.float32) / 255.0

    # columns 0 1 2 0*0 0*1 0*2 1*1 1*2 2*2
    d0 = data[:,0].reshape(-1,1)
    d1 = data[:,1].reshape(-1,1)
    d2 = data[:,2].reshape(-1,1)

    #return data
    result = np.hstack((data, d0*d0, d0*d1, d0*d2, d1*d1, d1*d2, d2*d2))
    
    return result

def find_path_center(edge_img, y_half, x_half):
    # left and right row vectors from centerpoint
    left_row = edge_img[y_half,:x_half]
    right_row = edge_img[y_half,x_half+1:]

    print len(left_row), x_half

    # find value going outwards that is zero/edge(255)x
    i = x_half-1
    left_row_path_location = 0
    while True:
        if left_row[i] == 255:
            left_row_path_location = i
            break
        i-=1
    print 'left row path location is: ',left_row_path_location
    j = 0
    right_row_path_location = 0
    while True:
        if right_row[j] == 255:
            right_row_path_location = j
            break
        j+=1
    print 'right row path location is: ',right_row_path_location

    # location variables are pixels on row vector
    path_width = abs((x_half + right_row_path_location)-left_row_path_location)
    path_center = math.ceil(path_width/2)

    #index of center of given row
    img_center = left_row_path_location + path_center 
    return int(img_center)

pklfile = open('classifier.pkl','rb')
data1 = pickle.load(pklfile)

filename = 'odroid/outside2.jpeg'
test_rgb = cv2.imread(filename)
result = data1.predict(make_input(test_rgb.reshape(-1, 3)))
    
img = result.reshape(test_rgb.shape[:2]) * 255

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# view the two images and waitkey to exit
cv2.imshow('win', np.hstack((test_rgb, img_color)))
while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass

# canny egde detection to help with path realization
edges = cv2.Canny(img_color,100,200)
cv2.imshow('win', edges)
while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass

center = find_path_center(edges,y_half, x_half)
radius = 20
cv2.circle(img_color, (center, y_half), radius, (0,0,255))
cv2.imshow('win', img_color)
while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass
