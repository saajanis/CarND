import math
from os import listdir
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary import image

def load_images(IMG_DIR):
    images = []
    for image_path in listdir(IMG_DIR):
        if image_path == '.DS_Store':
            continue
        img = cv2.cvtColor(cv2.imread(IMG_DIR + image_path), cv2.COLOR_BGR2RGB)
        images.append(img)
    return images

def show_images(images, make_random=False, fig_title='Default title', CMAP=None):
    pass
#     count = len(images)
#     col_count = 5
#     rows = math.ceil(float(len(images)) / float(col_count))
#     fig = plt.figure(figsize=(4*int(col_count),2*rows))
#     fig.suptitle(fig_title, fontsize=16)
#     
#     for i in range(count):
#         image = images[i]
#         ax1 = fig.add_subplot(rows,col_count,i+1)
#         #ax1.set_title(labels[i] + ' occurences', fontsize=8)
#         ax1.imshow(image, cmap=CMAP)
    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    
import cv2
import numpy as np

# Camera calibration
OBJ_POINTS = np.zeros((6*9, 3), np.float32)
OBJ_POINTS[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # x,y coordinates
def calibrate_camera(images):
    imgpoints = []
    objpoints = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        image = cv2.drawChessboardCorners(image, (9,6), corners, ret)
        
        if (ret == True):
            imgpoints.append(corners)
            objpoints.append(OBJ_POINTS)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return (mtx, dist)

def undistort_image(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)

calibration_images = load_images('./camera_cal/')
#show_images(calibration_images, fig_title='Original Calibration Images')

(MTX, DIST) = calibrate_camera(calibration_images)
#show_images(undistorted_images, fig_title='Undistorted Images')

# Transform color and gradient
# TODO(saajan): See if this needs improvement
def transform_sobel_s_(image, s_thresh=(170, 255), sx_thresh=(20, 100)):
    image = np.copy(image)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hsv[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return combined_binary

# Perspective transform

SRC = np.float32([[590, 460], [330, 670], [1060, 670], [700, 460]])
DST = np.float32([[330, 0], [330, 670], [1060, 670], [1060, 0]])
M = cv2.getPerspectiveTransform(SRC, DST)
M_inverse = cv2.getPerspectiveTransform(DST, SRC)
def transform_perspective(image):
    [y_orig, x_orig] = image.shape[:2]
    warped = cv2.warpPerspective(image, M, (x_orig, y_orig), flags=cv2.INTER_LINEAR)
#     # Testing by drawing a rectangle
#     image = cv2.polylines(image, np.int32([SRC.reshape((-1,1,2))]), True, (0,255,255))
#     show_images([image], fig_title='Rect Image')
    return warped

def transform_perspective_reverse(image, original_image = None):
    [y_orig, x_orig] = image.shape[:2]
    warped = cv2.warpPerspective(image, M_inverse, (x_orig, y_orig), flags=cv2.INTER_LINEAR)
    return warped

from moviepy.video.io.bindings import mplfig_to_npimage

# Locate lane lines
def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def find_window_centroids(image, window_width, window_height, margin):
    
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))

    return window_centroids

LEFT_FIT = None
RIGHT_FIT = None
def locate_lane_lines(image, original_image = None):
    global LEFT_FIT
    global RIGHT_FIT
    
    out_img = np.dstack((image, image, image))*255
    window_img = np.zeros_like(out_img)
        
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    nonzero_img_mask = np.zeros_like(image)
    nonzero_img_mask[nonzeroy, nonzerox] = 1

    # Set the width of the windows +/- margin
    margin = 150
    # Set minimum number of pixels required to recenter window
    minpix = 50
    # Set minimum number of pixels required to reconsider last image's fits
    # TODO(saajan): Rethink the value of this param
    min_pix_full_image = 50000
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Evaluate if the last fit can be used
    if(LEFT_FIT is not None and RIGHT_FIT is not None):
        left_lane_inds_temp = ((nonzerox > (LEFT_FIT[0]*(nonzeroy**2) + LEFT_FIT[1]*nonzeroy + LEFT_FIT[2] - margin)) & (nonzerox < (LEFT_FIT[0]*(nonzeroy**2) + LEFT_FIT[1]*nonzeroy + LEFT_FIT[2] + margin))) 
        right_lane_inds_temp = ((nonzerox > (RIGHT_FIT[0]*(nonzeroy**2) + RIGHT_FIT[1]*nonzeroy + RIGHT_FIT[2] - margin)) & (nonzerox < (RIGHT_FIT[0]*(nonzeroy**2) + RIGHT_FIT[1]*nonzeroy + RIGHT_FIT[2] + margin)))  
        # Again, extract left and right line pixel positions
        leftx_temp = nonzerox[left_lane_inds_temp]
        rightx_temp = nonzerox[right_lane_inds_temp]
        
        if (len(leftx_temp) + len(rightx_temp) < min_pix_full_image):
            LEFT_FIT = None
            RIGHT_FIT = None
        
    if(LEFT_FIT is None and RIGHT_FIT is None):
        # window settings
        window_width = 50 
        window_height = 80 # Break image into 9 vertical layers since image height is 720
        
        window_centroids = find_window_centroids(image, window_width, window_height, margin)

        left_img_mask = np.zeros_like(image)
        right_img_mask = np.zeros_like(image)
        # If we found any window centers
        if len(window_centroids) > 0:
            # Go through each level and draw the windows     
            for level in range(0,len(window_centroids)):
                # Window_mask is a function to draw window areas
                l_mask = window_mask(window_width,window_height,image,window_centroids[level][0],level)
                r_mask = window_mask(window_width,window_height,image,window_centroids[level][1],level)
                
                # Accumulate window areas
                left_img_mask = left_img_mask | l_mask
                right_img_mask = right_img_mask | r_mask
         
        # Final non-zero pixels in window areas    
        left_img_mask = left_img_mask & nonzero_img_mask
        right_img_mask = right_img_mask & nonzero_img_mask
        
        nonzero_left = left_img_mask.nonzero()
        nonzero_right = right_img_mask.nonzero()
        
        # Fit a second order polynomial to each
        LEFT_FIT = np.polyfit(nonzero_left[0], nonzero_left[1], 2)
        RIGHT_FIT = np.polyfit(nonzero_right[0], nonzero_right[1], 2)
            
    if(LEFT_FIT is None and RIGHT_FIT is None):
        return image
    else:
        # Visualizing
        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        left_fitx = LEFT_FIT[0]*ploty**2 + LEFT_FIT[1]*ploty + LEFT_FIT[2]
        right_fitx = RIGHT_FIT[0]*ploty**2 + RIGHT_FIT[1]*ploty + RIGHT_FIT[2]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        
        reverse_transformed_mask = transform_perspective_reverse(window_img)
        #reverse_transformed_mask = np.dstack((reverse_transformed_mask, reverse_transformed_mask, reverse_transformed_mask))*255
        
        result = cv2.addWeighted(original_image, 1, reverse_transformed_mask, 0.3, 0) 
        
        return result




####################################
from functools import partial

#Pipeline
test_images = load_images('./test_images/')

PIPELINE_FUNCS = [partial(undistort_image, mtx = MTX, dist = DIST), transform_sobel_s_, transform_perspective,
                 locate_lane_lines]

transformed_images = test_images
for func in PIPELINE_FUNCS:
    transformed_images = [func(image) for image in transformed_images]

show_images(transformed_images, fig_title='Transformed Images')