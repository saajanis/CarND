
# coding: utf-8

# In[1]:


class MacOSFile(object):
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size

def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))


# In[2]:


import math
import matplotlib.pyplot as plt
plt.ion()

def show_images(images, make_random=False, fig_title='Default title', CMAP=None):
    count = len(images)
    col_count = 5
    rows = math.ceil(float(len(images)) / float(col_count))
    fig = plt.figure(figsize=(4*int(col_count),2*rows))
    fig.suptitle(fig_title, fontsize=16)
    
    for i in range(count):
        image = images[i]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax1 = fig.add_subplot(rows,col_count,i+1)
        #ax1.set_title(labelssamples[i] + ' occurences', fontsize=8)
        ax1.imshow(image, cmap=CMAP)


# In[3]:


# lesson_functions

import numpy as np
import cv2
from skimage.feature import hog
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
# Define a function to compute color histogram features 
# TODO: NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


#Feature extraction
def extract_features(img, cspace='RGB', spatial_size=(32, 32),
                        hist_bins=32, hist_range=(0, 256)):
    image = cv2.resize(img, (64, 64))

    # apply color conversion if other than 'RGB'
    if cspace != 'RGB':
        if cspace == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)/255  
        elif cspace == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV).astype(np.float32)/255 
        elif cspace == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float32)/255 
        elif cspace == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV).astype(np.float32)/255  
    else: 
        feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255    

    # Apply bin_spatial() to get spatial color features
    spatial_features = bin_spatial(feature_image, size=spatial_size)
    # Apply color_hist() also with a color space option now
    hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)

    ch1 = feature_image[:,:,0]
    ch2 = feature_image[:,:,1]
    ch3 = feature_image[:,:,2]
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False).ravel()
    hog_features = np.hstack((hog1, hog2, hog3))

    all_features = np.hstack((spatial_features, hist_features, hog_features))

    
    features = np.array(all_features).astype(np.float64)
        
    return features


# In[4]:


# Find_cars assisting functions

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    
    return img

def is_similar_prev_bboxes(bbox, prev_bboxes_list, scale, img):
    if len(prev_bboxes_list) == 0:
        return True
    for prev_bboxes in prev_bboxes_list:
        prev_bboxes_scale = prev_bboxes[scale]
        bb_intersection_over_union(bbox, prev_bboxes_scale, img)
        
        #return False if not similar enough to at least 1 in prev_bboxes_scale
        
        
    return True   
        
        
def bb_intersect(boxA, boxB):
    return not (boxA[1][0] < boxB[0][0] or boxA[0][0] > boxB[1][0] or boxA[1][1] < boxB[0][1] or boxA[0][1] > boxB[1][1])       

def bb_intersection_over_union(boxA, boxB_list, img):
    if len(boxB_list) == 0:
        return True
    
    for boxB in boxB_list:
        ##
        draw_img = np.copy(img)
        cv2.rectangle(draw_img, boxA[0], boxA[1], (255,0,0), 6) #blue
        cv2.rectangle(draw_img, boxB[0], boxB[1], (0,255,0), 6) #green
        x = 0
        
        if not bb_intersect(boxA, boxB):
            return False
        
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0][0], boxB[0][0])
        yA = max(boxA[0][1], boxB[0][1])
        xB = min(boxA[1][0], boxB[1][0])
        yB = min(boxA[1][1], boxB[1][1])
        # compute the area of intersection rectangle
        interArea = (xB - xA + 1) * (yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[1][0] - boxA[0][0] + 1) * (boxA[1][1] - boxA[0][1] + 1)
        boxBArea = (boxB[1][0] - boxB[0][0] + 1) * (boxB[1][1] - boxB[0][1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        ##
        x = 0
        
        
    
    return True # remove this
#     # No match found
#     return False


# In[5]:


from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import glob
import time
import pickle

#TODO: Shuffle dataset
# Read in car and non-car images
car_images = glob.iglob('./data/all_data/vehicles/**/*.png', recursive=True)
not_car_images = glob.iglob('./data/all_data/non-vehicles/**/*.png', recursive=True)
cars = []
notcars = []
for image in car_images:
    cars.append(image)
for image in not_car_images:
    notcars.append(image)

# cars = cars[:200]
# notcars = notcars[:200]
        
# Define the labels vector
y = np.hstack((np.ones(len(cars)), np.zeros(len(notcars))))

### TODO: Tweak these parameters and see how the results change.
color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 10  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)
histbin = 32 # Number of histogram bins
histrange = (0, 256)
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()
X_scaler = StandardScaler()


# # Get all features
# all_features = []
# for img in (cars+notcars):
#     all_features.append(extract_features(cv2.imread(img), cspace=color_space, spatial_size=spatial_size, 
#                      hist_bins=histbin, hist_range=histrange))

# # Save features
# pickle.dump(all_features, open('all_features_3', 'wb'))
#Load features
all_features = pickle.load(open('all_features_3', 'rb'))

# Fit a per-column scaler
X_scaler.fit(all_features)
# Apply the scaler to X
scaled_features = X_scaler.transform(all_features)

print(len(scaled_features))
 
rand_state = np.random.randint(0, 10.0)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, y, test_size=0.2, random_state=rand_state)
 
svc = LinearSVC(C=100)
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
 
print(round(t2-t, 2), 'Seconds to train SVC...')
 
    
    
# #Save state
# model_pickle = {}
# model_pickle["svc"] = svc
 
# # Save model
# pickle.dump(model_pickle, open('pickled_model_3', 'wb'))
# Load model
loaded_svc_model = pickle.load(open('pickled_model_3', 'rb'))

svc = loaded_svc_model["svc"]

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


# In[17]:


from scipy.ndimage.measurements import label

ystart = 400
ystop = 656
xstart = 600
window_scales = [0.5, 1.0, 1.5, 2.0]
overlap_thresholds_for_scales = {0.5: 1, 1.0: 1, 1.5: 1, 2.0: 1} #was {0.5: 6, 1.0: 1, 1.5: 2, 2.0: 1}
min_prev_bboxes_for_match = {0.5: 2, 1.0: 1, 1.5: 1, 2.0: 1}
frame_count = 0
#cached_bounding_boxes = {}
cached_bounding_boxes = pickle.load(open('cached_bounding_boxes', 'rb'))
    
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, xstart, window_scales, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, cspace=color_space):
    global frame_count
    global cached_bounding_boxes
    global overlap_thresholds_for_scales
    global min_prev_bboxes_for_match
    cached_bounding_boxes_for_frame = {}
    for scale in window_scales:
        cached_bounding_boxes_for_frame[scale] = []
    
    draw_img = np.copy(img)
    heat_img = np.zeros_like(img[:,:,0]).astype(np.float)
    
    
    
    
#     # Compute draw_img and heat_img from scratch
#     orig_img_tosearch = img[ystart:ystop,xstart:,:]
#     for scale in window_scales:
#         if cspace=='HSV':
#             ctrans_tosearch = cv2.cvtColor(orig_img_tosearch, cv2.COLOR_RGB2HSV).astype(np.float32)/255
#         if scale != 1.0:
#             imshape = ctrans_tosearch.shape
#             ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))



#         ch1 = ctrans_tosearch[:,:,0]
#         ch2 = ctrans_tosearch[:,:,1]
#         ch3 = ctrans_tosearch[:,:,2]

#         # Define blocks and steps as above
#         nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
#         nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
#         nfeat_per_block = orient*cell_per_block**2

#         # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
#         window = 64
#         nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
#         cells_per_step = 2  # Instead of overlap, define how many cells to step
#         nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
#         nysteps = (nyblocks - nblocks_per_window) // cells_per_step

#         # Compute individual channel HOG features for the entire image
#         hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
#         hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
#         hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

#         for xb in range(nxsteps):
#             for yb in range(nysteps):
#                 ypos = yb*cells_per_step
#                 xpos = xb*cells_per_step
#                 # Extract HOG for this patch
#                 hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#                 hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#                 hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
#                 hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

#                 xleft = xpos*pix_per_cell
#                 ytop = ypos*pix_per_cell

#                 # Extract the image patch
#                 subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

#                 # Get color features
#                 spatial_features = bin_spatial(subimg, size=spatial_size)
#                 hist_features = color_hist(subimg, nbins=hist_bins)

#                 # Scale features and make a prediction
#                 all_features = np.hstack((spatial_features, hist_features, hog_features))
#                 features = np.array(all_features).astype(np.float64)
#                 scaled_features = X_scaler.transform([all_features])
#                 #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
#                 test_prediction = svc.predict(scaled_features)

#                 if test_prediction == 1:
#                     xbox_left = np.int(xleft*scale)
#                     ytop_draw = np.int(ytop*scale)
#                     win_draw = np.int(window*scale)
#                     bbox = ((xbox_left+xstart, ytop_draw+ystart), (xbox_left+win_draw+xstart,ytop_draw+win_draw+ystart))
#                     cv2.rectangle(draw_img,bbox[0],bbox[1],(0,0,255),6) 
#                     heat_img[ytop_draw+ystart:ytop_draw+win_draw+ystart, xbox_left+xstart:xbox_left+win_draw+xstart] += 1
#                     cached_bounding_boxes_for_frame[scale].append(bbox)

#     cached_bounding_boxes[frame_count] = cached_bounding_boxes_for_frame
    
    
    
    
    
    
    # Load draw_img and heat_img from cache
    cached_bounding_boxes_for_frame = cached_bounding_boxes[frame_count]
    for scale in window_scales:
        cached_bounding_boxes_for_scale = cached_bounding_boxes_for_frame[scale]
        threshold_scale = overlap_thresholds_for_scales[scale]
        heat_img_scale = np.zeros_like(img[:,:,0]).astype(np.float)
        prev_bboxes_list = [cached_bounding_boxes[x] for x in [i for i in range(frame_count-min_prev_bboxes_for_match[scale], frame_count)] if x in cached_bounding_boxes]

        
        for bbox in cached_bounding_boxes_for_scale:
            # Find if similar enough to all in prev_bboxes at this scale
            if is_similar_prev_bboxes(bbox, prev_bboxes_list, scale, np.copy(img)):
                cv2.rectangle(draw_img,bbox[0],bbox[1],(0,0,255),6) 
                heat_img_scale[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]] += 1
        
        heat_img_scale[heat_img_scale < threshold_scale] = 0 
        heat_img = heat_img + heat_img_scale
        
        
        
        
        
        
    heatmap = np.clip(heat_img, 0, 255)
    labels = label(heatmap)
    final_heat_img = draw_labeled_bboxes(np.copy(img), labels) 
    
    
    frame_count = frame_count+1
    return (final_heat_img, draw_img)


# In[18]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image):
    (heat_img, draw_img) = find_cars(image, ystart, ystop, xstart, [window_scales[3]], svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, histbin)
    return heat_img

output_path = 'output_video.mp4'

frame_count = 0
output_path = 'heat_img_scale_05.mp4'
project_video = VideoFileClip("project_video.mp4")#.subclip(24,25)
project_clip = project_video.fl_image(process_image)
project_clip.write_videofile(output_path, audio=False)

pickle.dump(cached_bounding_boxes, open('cached_bounding_boxes', 'wb'))


# In[ ]:




