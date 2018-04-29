[//]: # (Image References)

[image1]: ./writeup_images/VGG.jpg
[image2]: ./writeup_images/fcn.jpg
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./writeup__supporting_data/labels_histogram.png "Labels histogram"
[image10]: ./writeup__supporting_data/sample_images_training.png "Sample images training"
[image11]: ./writeup__supporting_data/original_normalized.png
[image12]: ./writeup__supporting_data/sample_images_training_cropped.png "Sample images training cropped"
[image13]: ./writeup__supporting_data/sample_images_training_cropped_normalized.png
[image14]: ./writeup__supporting_data/sample_images_training_cropped_grayscaled.png
[image15]: ./writeup__supporting_data/sample_images_training_cropped_grayscaled_normalized.png
[image16]: ./writeup__supporting_data/original_data_plot.png
[image17]: ./writeup__supporting_data/original_normalized_data_plot.png
[image18]: ./writeup__supporting_data/cropped_data_plot.png
[image19]: ./writeup__supporting_data/cropped_grayscale_data_plot.png
[image20]: ./writeup__supporting_data/cropped_normalized_data_plot.png
[image21]: ./writeup__supporting_data/cropped_grayscaled_normalized_data_plot.png
[image22]: ./writeup__supporting_data/sample_test_images.png


# Overview
The goal is to create a Fully Convolutional Network (FCN) to be able to label each pixel in an image as road/non-road.



We're given a frozen, pre-trained VGG16 model (a fully concolution version with a 1 by 1 convolution to replace the fully connected layer) represented below.
![""][image1]


The model is then connected to a few upsampling layers based on the FCN-8 architecture represented below.
![""][image2]

Another 1x1 convolution will then reduce the depth of the output of frozen VGG16 from 4096 to 2 (num_classes). 

The upsampling will keep this depth, but increase the height and width to the original image's dimensions.

The skip connection layers add together the pooling layers from the VGG16 network to these upsampled layers to get more spatial information from previous layers to increase accuracy.


## Interesting points about the implementation

The network recognizes two classes - road/non-road. 


The kernels used in all tf.layers.conv2d\* layers were all:
- initialized with a tf.random_normal_initializer(stddev=0.01)
- regularized using a tf.contrib.layers.l2_regularizer(scale=1e-3) 


The loss function is a sum of cross entropy losses (with an optimization objective of reducing the mean) and the regularization terms mentioned previously (with the optimization objective of reducing their sum).


The loss function is fed to an Adam Optimizer with a learning rate of 0.0009


The network was trained for 50 epochs with a batch size of 5.


The loss reduces roughly as: 
- ~1.5   (epoch=1)
- ~0.07  (epoch=11)
- ~0.05  (epoch=21)
- ~0.04  (epoch=31)
- ~0.03  (epoch=41)
- ~0.027 (epoch=50)


The images from test run are present in the runs directory.

The output from the latest run of the program is in output.txt