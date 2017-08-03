[//]: # (Image References)

[image1]: ./writeup_data/calibration_images.png "Calibration"
[image2]: ./writeup_data/test_images.png "Test"
[image3]: ./writeup_data/color_gradient_transform.png "colorGradientTransform"
[image4]: ./writeup_data/perspective_transform.png "PerspectiveTransform"
[image5]: ./writeup_data/lane_lines.png "lane_lines"
[image6]: ./writeup_data/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### The code for the project is contained in the iPython notebook located [here](https://github.com/saajanis/CarND/blob/master/CarND-PreProject4/AdvancedLaneLines/pipeline.ipynb)

---
#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

Here it is!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `OBJ_POINTS` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 


### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here are the undistorted versions of the provided camera calibration images:
![Calibration][image1]

#### 2. Test images

I selected a total of 13 test images (added some to the ones provided) to construct my pipeline. Some of these images are interesting and challenging to the pipeline because:

* Some of them have shadows.
* Some of them have lane lines missing in the bottom of the frame (due to the position of the vehicle at that time).
* Patch work on the roads - different appearance of the road etc.

Here are the images: 
![Test][image2]

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (code in cell 3).  

* I found that sobel transform in x direction gave the best results while trying to detect lines. I used the threshold range of <b>(90, 255)</b> while transforming the images.
* I found that the S channel in the HLS transform gave me the best results while trying to detect the lane line of different colors. I used the range <b>(15, 100)</b> while trying to transform.

I took a sum of the pixels in the masks generated after the two steps, which gave me these color and gradient transformed images:

![colorGradientTransform][image3]

#### 4. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I examined a test image and chose certain points on the image to include the lane lines (code in cell 4).

I chose the SRC and DST points to generate a perspective transform using the cv2.getPerspectiveTransform(SRC, DST) function  I chose to hardcode the source and destination points in the following manner:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 460      | 330, 0        | 
| 330, 670      | 330, 670      |
| 1060, 670     | 1060, 670     |
| 700, 460      | 1060, 0       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![PerspectiveTransform][image4]

#### 5. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Cell 5 contains the code for identifying the lane lines. I used the sliding windows approach using convolutions to detect the lane lines.

* I chose a window 50 px for convolutions, a margin of 175px * 2, for detecting non-zero pixels and a minimum of 50 non-zero px to allow readjusting the window centroid.
* Also, I chose a value of 65000 px as the min number of non-zero pixels required within the margin from the previously fit polynomials in order to reuse it for the current frame. Of course, while generating images in the next step, the polynomials were computed separately.

The fit is shown in the images in step 7.

#### 6. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Towards the end of cell 5, left_curverad and right_curverad respresent the radius of curvature of the left and right lane lines respectively (ideally, the should be the same if the lines are parallel and closeness of these values will signify the robustness of our pipeline). 

I also computed the offset of the vehicle form the centre of the road by computing the diference between the centre of the camera's frame and the centre of the detected lane lines close to the car.

The values computed are shown in the images in step 7.

#### 7. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Some interesting this that I observed:
* The algorithm had a particularly difficult time detecting lane lines when the images were missing lines in the bottom of the frame. Notice the 10th and 11th image and where the curve starts (those are the points with the most pixels in that column od the image).
* The algorithm sometimes started off fine but then merged with the other lane's lines. 

Here are the images :

![lane_lines][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/CVde9Pet_UA)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I will discuss the issues mentioned in step 7 here:
* For the first one, I could experiment with initalizing the window cetroids from the top or even middle of the frame (or a combination of top, down and middle and select one which gives the highest confidence) to get over the problem of missing lane lines in some part of the frame sometimes.
* For the second point, I could experiment with making the window height bigger or increasing the number of pixels required to recenter the windows or find a way to reuse the last fit when computed fit is erroneous like that. Another solution would be to make the algorithm smart enough to detect that once a polynomial is fit for a lane line with high confidence, to not look more than half way through from the starting point to the other lane line to prevent merging and increase the chances of the actual lane to be detected later on by a different window.