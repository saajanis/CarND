**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image11]: ./examples/car_not_car.png
[image1]: ./writeup_data/1.png "Calibration"
[image2]: ./writeup_data/2.png "Test"
[image3]: ./writeup_data/3.png "colorGradientTransform"
[image4]: ./writeup_data/4.png "PerspectiveTransform"
[x11]: ./writeup_data/x11.png "PerspectiveTransform"
[x12]: ./writeup_data/x12.png "PerspectiveTransform"
[x21]: ./writeup_data/x21.png "PerspectiveTransform"
[x22]: ./writeup_data/x22.png "PerspectiveTransform"
[x31]: ./writeup_data/x31.png "PerspectiveTransform"
[x32]: ./writeup_data/x32.png "PerspectiveTransform"
[y1]: ./writeup_data/y1.png "PerspectiveTransform"

[image5]: ./writeup_data/lane_lines.png "lane_lines"
[image6]: ./writeup_data/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### The code for the project is contained in the iPython notebook located [here](https://github.com/saajanis/CarND/tree/master/CarND-Project5/CarND-Vehicle-Detection/Pipeline3.ipynb)

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the third code cell of the IPython notebook in the method get_hog_features(...).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image11]

I then tried out the values I already had from the codelabs in the lessons:
* orient = 9
* pix_per_cell = 8
* cell_per_block = 2

which seemed to work well.

####2. Explain how you settled on your final choice of HOG parameters.

I did not have to explore much in tuning the various parameters for hog since after using the ones I had been using in the codelabs in the lessons (alongside other features), I got a test set accuracy of 99.86% on my SVM, which was good enough for me to proceed.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In addition to HOG features, I also decided to use binned color features, as well as histograms of color features. All these features are extracted for each sub-frame in cell 3's extract_features(...) method.

I trained a linear SVM using the features extracted above. The code is in cell 3.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to test out the following values of scales which correspond to different window sizes: 0.1, 1.0, 1.5, 2.0. For each scale value, there are corresponding values for min overlap required in the heat image, min number of matching boxes in the previous detections required for the current frames bounding box to be accepted, and min threshold for the extent of overlap with the previous windows. These values are coded in overlap_thresholds_for_scales, min_prev_bboxes_for_match and min_iou_for_scales dictionaries respectively.

[Here's](https://youtu.be/es4y3ru9Wv8) a video of the raw window detections (before generating a heat image) for these scales and parameter values chosen.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/kH_MCrb8sHs)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Detecting bounding boxes:
- The pipeline detects overlap between detected bounding boxes and filters out areas with at less than 3 ovelapping bounding boxes in a 'heat_img'.
- These bounding boxes are only detected for a portion of the frame where the cars can appear.
- The pipeline stores all the bounding boxes detected in the previous n frames where n is different for different scale sizes, whether or not they were rejected (i.e. they were shown in the final video).
- For each bounding box that the pipeline decides to show in the final video (more discussion later), the pipeline detects similar bounding boxes in the detections in last *25 frames* and averages over their size to get a smooth transition in the bounding boxes frame over frame.

Rejecting false positives:
- The averaging function (in get_average_bbox(...)) uses the area intersected (in bb_intersection_over_union(...) method) and checks that it is above a certain ratio of similarity to previous bounding boxes (or lack thereof) while detecting similar bounding boxes and averages over them.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected where heat was greater than 3 (corresponding to 3 overlaps).  

Here's an example result showing the heatmap from a series of frames of video and the corresponding bounding boxes detected:

### Here are three heatmaps and their corresponding bounding box detections:

![alt text][x11]
![alt text][x12]
![alt text][x21]
![alt text][x22]
![alt text][x31]
![alt text][x32]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

All of these are things I would've tried to make the pipeline better if I had more time:
- Although my SVM had an accuracy of 99.77%, it feels weird that there are still so many false positive detections. I want to evaluate what's different between the images in my training data and the sample video to make the SVM perform better on the video.

- This pipeline is likely to fail on videos taken in the dark since the frames are not similar to the training examples (at least based on the features I extracted). I want to find a way to make it insensitive to lighting conditions on the road through transforming to another color space or normalizing the attributes of the image such as saturation.
