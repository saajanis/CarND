
[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./writeup__supporting_data/NVIDIA_CNN.png "NVIDIA_CNN"
[image10]: ./writeup__supporting_data/track1_forward.png "Sample images training"
[image11]: ./writeup__supporting_data/track1_forward_backward.png
[image12]: ./writeup__supporting_data/track1_forward_backward_ForwardSlow.png "Sample images training cropped"
[image13]: ./writeup__supporting_data/track1_forward_backward_ForwardSlow_forwardCorrectional.png
[image14]: ./writeup__supporting_data/track1_forward_backward_ForwardSlow_patches.png
[image15]: ./writeup__supporting_data/sample_center_image.jpg
[image16]: ./writeup__supporting_data/sample_left_image.jpg
[image17]: ./writeup__supporting_data/sample_right_image.jpg



[image18]: ./writeup__supporting_data/cropped_data_plot.png
[image19]: ./writeup__supporting_data/cropped_grayscale_data_plot.png
[image20]: ./writeup__supporting_data/cropped_normalized_data_plot.png
[image21]: ./writeup__supporting_data/cropped_grayscaled_normalized_data_plot.png
[image22]: ./writeup__supporting_data/sample_test_images.png


## Project 3 writeup
Here is a link to my [project repository](https://github.com/saajanis/CarND/tree/master/CarND-PreProject3/CarND-Behavioral-Cloning-P3) which contains the corresponding output too.

### Rubric Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

#### 1. Files Submitted & Code information

Submission includes all required files and can be used to run the simulator in autonomous mode. My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Instructions for running

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model.

Submission includes functional code using the Udacity provided simulator and the original drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.h5

#### 3. Model Architecture and Training Strategy

I used the architecture used by NVIDIA in their self-driving cars that was shown in the lessons. Here are the details of the architecture:

The train_model() method takes a pre-existing model (or creates a new one if none is supplied) and trains the previous model on the newly supplied data to output a new model. This helped me in snapshotting the models and try different strategies for the kind of data that works.

I used a CNN mixed with maxpooling. This image (taken from their [blog](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)) underlying model architecture:

![NVIDIA_CNN][image9]

I chose a CNN based network architecture because they usually perform bettwr in tasks involving images.

* For normalization, I use the recommended method to convert the pixel values to normalized, zero mean values. The data is normalized in the model using a Keras lambda layer.

* I used a 2x2 stride for where the number of channels is seen as changing between the convolutional layers.

* The model includes RELU layers to introduce nonlinearity.

* The model contains a dropout layer in order to reduce overfitting.

* 20% of the original data was set aside to ensure that the model does not overfit.

* Successful test on the simulator ensured that the model, in fact, does generalize.

* The model used an adam optimizer, so the learning rate was not tuned manually.

* The optimization function was chosen to be *minimizing mean square error* for the predicted steering angle.

#### 4. Training data collection

While collecting data, my initial intuition was to collect good driving behaviour when everything was going right (car was in the middle of the road) as well as behavior to recover when the car starts going off track.

Here are the different kinds of datasets (from track 1) that I had for training and what I found about them:

 * Driving forward normally (center of the road, fairly good speed):  
 
 ![track1_forward][image10]
 
 Car drove nicely in the simulator but went off-track pretty frequently.
 
 <br/><br/>
  * Driving forward and then backward, normally:  
 
 ![track1_forward_backward][image11]
 Car drove nicely in the simulator but went off-track relatively less frequently.
 
 <br/><br/>
  * Driving forward , backward normally and then forward really slowly:  
 
 ![track1_forward_backward_ForwardSlow][image12]
 Car drove nicely in the simulator and went off-track very infrequently drove off-track consistently at some patches (about 3) of the track.
 
 <br/><br/>
  * Driving forward , backward normally and then forward really slowly. Then, I tried to add some data for recovering from when the car goes off-track:  
 
 ![track1_forward_backward_ForwardSlow_forwardCorrectional][image13]
 
 There wasn't any overfitting here as in the previous models, but the car drove really erratically. The increase in validation error shows the same fact. I think it is because of the way I collected data. I would drive the car to the side of the road, start recording and comeback to the center very aggressively - maybe I should've come back to the center slowly, like in a real situation where I am correcting the car going off-course. I considered recollecting this data with the improved strategy, but the next model just worked - so I'll ignore this model/data.
 
 <br/><br/>
  * Driving forward , backward normally and then forward really slowly. Then, I tested out this data in the simulator and collected more data for the patches where the car was going off track and trained some more on that data:  
 
 ![track1_forward_backward_ForwardSlow_patches][image14]
 I collected more data specifically for the patches where the car drove off road and trained it into the last to previous model and voila, it gave me a model that drove the car nicely through the whole track! The output corresponds to driving with this model.

<br/><br/>


#### 5. Creation of the Training Set and data augmentation

For each of the datasets collected above, I applied the following transformations on the data before feeding it to the network:

* A sample image for the car in the centre of the road looked like this:
 ![sample_center_image][image15]
 
 And the left and right samples (which simulate an image taken from cameras mounted on the left and right of the car's dashboard) look like this:
 ![sample_left_image][image16]
 <br/>
 ![sample_right_image][image17]
 
 While the steering angle for the image at the centre stays as recorded - for the left and right images, the target steering angles are added and subtracted respectively with a *correction of 0.25* so we can treat them as if they were the centre image.

* I then applied flipping to each image to generate a laterally inverted image (to simulate driving in the opposite direction) which will help the model generalize.

* Also, built into the model is a cropping mechanism that crops out the top 70 pixels (the sky, shrubs etc.) and bottom 25 pixels (hood of the car) from each image in the training and testing dataset. They'll confuse the model rather than help and provide no useful information while making gthe decision if every image has those pixels.

* The model also has a layer to normalize the data to zero mean and small values.
 

#### 6. Results

The result is the following video of the car driving itself successfuly on track 1 (click on the thumbnail):

[![Track_1_youtube_video](https://i.ytimg.com/vi/16tdcVb8rtE/2.jpg?time=1500788879230)](https://youtu.be/16tdcVb8rtE)


```python

```
