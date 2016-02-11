import cv2
import numpy as np
import glob, os
import sys
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

    #print len(left_row), x_half

    # find value going outwards that is zero/edge(255)x
    i = x_half-1
    left_row_path_location = 0
    while True:
        if left_row[i] == 255:
            left_row_path_location = i
            break
        i-=1
    #print 'left row path location is: ',left_row_path_location
    j = 0
    right_row_path_location = 0
    while True:
        if right_row[j] == 255:
            right_row_path_location = j
            break
        j+=1
    #print 'right row path location is: ',right_row_path_location

    # location variables are pixels on row vector
    path_width = abs((x_half + right_row_path_location)-left_row_path_location)
    path_center = math.ceil(path_width/2)

    #index of center of given row
    img_center = left_row_path_location + path_center 
    return int(img_center)


def predict(clf, rgb):

    assert rgb.shape[-1] == 3
    
    orig_shape = rgb.shape[:-1]
    input_data = make_input(rgb.reshape(-1, 3))

    result = ( np.dot(input_data, clf['coef']) + clf['intercept'] > 0.0 ).astype(np.uint8)
    
    result = result.reshape(orig_shape)

    return result

clf = np.load('the_classifier.npz')

print 'loaded classifier'
filename = 'test2.jpg'

test_rgb = cv2.imread(filename)
result = predict(clf, test_rgb)

img = result * 255
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

#############################
# 0 1 2 ... w-1
xrng = np.arange(img.shape[1], dtype=np.float32)

accum = result.astype(np.float32) * xrng[None,:]
ctrx = (accum.sum(axis=1) / result.sum(axis=1)).astype(np.int32)
ctry = np.arange(img.shape[0])

points = np.hstack((ctrx.reshape(-1,1), ctry.reshape(-1,1)))
print points.shape, points.dtype

edge_padded = np.hstack((
    np.zeros((img.shape[0], 1), dtype=np.uint8),
    result,
    np.zeros((img.shape[0], 1), dtype=np.uint8)))

left_diff = cv2.absdiff(edge_padded[:,1:-1], edge_padded[:,:-2])
right_diff = cv2.absdiff(edge_padded[:,2:], edge_padded[:,1:-1])
diff = np.maximum(left_diff, right_diff)

# pre-initialize guess at path midpint to center of image
mid = img.shape[1]/2
xvals = []

# start at bottom of the image
for y in range(img.shape[0]-1, -1, -1):

    # look at all nonzero pixels in the diff image left of current midpoint
    left = np.nonzero(diff[y][:mid])[0]

    # look at all nonzero pixels in the diff image right of the current midpoint
    right = np.nonzero(diff[y][mid:])[0] + mid

    # average the biggest left x coordinate with the smallest right x coordinate
    mid = (left[-1] + right[0])/2

    # append this
    xvals.append(mid)

ctrx = np.array(xvals[::-1])

points2 = np.hstack((ctrx.reshape(-1,1), ctry.reshape(-1,1)))

mid_y = (2*img.shape[0])/5
end_y = (3*img.shape[0])/5
print mid_y
new_points = points2[mid_y:end_y]
x_avg, y_avg = np.mean(new_points, axis=0)
print 'average is', x_avg
print (x_avg/640)*180


#cv2.polylines(test_rgb, [points], False, (0,127,255), 1, cv2.LINE_AA)
cv2.polylines(test_rgb, [points2], False, (0,0,255), 1, cv2.LINE_AA)

#cv2.imshow('win', diff*255)
#while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass
####################################

  
# view the two images and waitkey to exit
print 'show img'
cv2.imshow('win', np.hstack((test_rgb, img_color)))
while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass

# canny egde detection to help with path realization
#print 'canny'
edges = cv2.Canny(img_color,100,200)
#cv2.imshow('win', edges)
#while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass

#print 'circles'
center = find_path_center(edges,y_half, x_half)
radius = 20
cv2.circle(img_color, (center, y_half), radius, (0,0,255))
cv2.imshow('win', img_color)
while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass

print 'center line is', x_avg
print 'center circle is', center

'''
for filename in glob.glob('*.[jJ][pP][eE][gG]'):
    test_rgb = cv2.imread(filename)
    
    result = predict(clf, test_rgb)
    
    img = result * 255
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    cv2.imshow('win', np.hstack((test_rgb, img_color)))
    cv2.waitKey(5)
    #while np.uint8(cv2.waitKey(5)).view(np.int8) < 0: pass
'''
print 'done baby'
