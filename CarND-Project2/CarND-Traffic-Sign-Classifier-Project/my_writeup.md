
[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
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


## Project 2 writeup
Here is a link to my [project code](https://github.com/saajanis/CarND/blob/master/CarND-Project2/CarND-Traffic-Sign-Classifier-Project/approach_eval.ipynb) with the corresponding output.

### Data set summary and exploration

#### 1. Data set(s) summary.

I used the numpy library conbined with python's native methods to calculate summary statistics of the traffic signs data set. Here are the summaries:

* Number of training examples = 34799
* Shape of an example image (in all datasets) = (32, 32, 3)
* Number of validation examples = 4410
* Number of testing examples = 12630
* Number of examples in my custom generated testing set = 5
* Number of classes = 43

#### 2. Data visualization.

##### Here's a histogram showing the distribution of the number of examples of each kind available in the datasets:
(if the x-axis labels look a bit off, find the nearest bar to the tick)
##### With an initial gaze, the number of examples of each type look pretty evenly distributed in each dataset (this is the last time we'll look at any statistic about the test set before we test our model). 

![labels_histogram][image9]

<br/><br/><br/>

##### Here are some randomly sampled images from the training dataset (arranged in the increasing order of their number of occurences) labelled with the name of the traffic sign and the number of times that sign appears in the training dataset:

![sample_images_training][image10]

<br/><br/><br/>


### Design and Test a Model Architecture


#### 1. Data set(s) preparation.

#### Transformations:
I decided to apply a few transformations to the data and test out my model's performance on each one of them. For each transformation, I'm including a few images after that transformation was applied to show what they look like (this is for general intuition -- I don't think it matters for how the neural net operates).
*Similar transformations were applied to images in each dataset*

**1. Normalization:**
* I took each dataset, computed the mean pixel value for all the images in the dataset and used it to normalize every pixel in every image in the dataset using the formula new_pixel_value = ((old_pixel_value - mean_pixel_value)/mean_pixel_value). This is with an expectation to make it easier for the model to work with the images since the mean of the pixels in the dataset is closer to zero now. Here's what the training dataset looks like now:

![original_normalized][image11]

<br/>
**2. Cropping the original images:**
* The provided dataset(s) have a 'coords' property which gives information about a bounding box around the traffic sign within the image. I cropped out the part specified by the bounding box in each of train, test and validation dataset and resized the image to 32*32 by padding it with black pixels. This should assist the neural network with translational variance in the images. For my testing dataset of 5 images, I purposefully did something similar by only taking a tightly bounded picture from the internet. Here's what the training dataset looks like now:

![sample_images_training_cropped][image12]

<br/>
**3. Normalization on the cropped images:**
* It's the same normalization, except that it is applied on the cropped images:

![sample_images_training_cropped_normalized][image13]

<br/>
**4. Grayscaling the cropped images:**
* Grayscaling will reduce the number of channels from 3 to 1 for each "pixel". This may work out if the model is too sensitive to slight variations in conditions under which images were shot etc.. 

![sample_images_training_cropped_grayscaled][image14]

<br/>
**5. Normalized grayscaled cropped images:**
* This is the same kind of normalization applied to the previous dataset. 

![sample_images_training_cropped_grayscaled_normalized][image15]

**Other things I would've tried if I had more time:**
* I was very curious to explore how generating more training images with the signs rotated by varying degrees would impact model performance. Sub-sampling takes care of translational variance but the model may still be sensitive to rotation.

<br/><br/><br/>


#### 2. Model architecture.

I used a CNN mixed with maxpooling. My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 for RGB image / 32x32x1 for grayscale | 
| Convolution 5x5/5x5   | 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 14x14x6    |
| Convolution 5x5	    | 1x1 stride, SAME  padding, outputs 10x10x6 	|
| Max pooling	      	| 2x2 stride, VALID padding, outputs  5x 5x6    |
| Flatten	      	    | outputs 400                                   |
| Fully connected		| outputs 120       							|
| RELU					|												|
| Fully connected		| outputs 84        							|
| RELU					|												|
| Fully connected		| outputs 43        							|
| Logit          		| outputs 43        							|
| Softmax				| outputs 43        							|
 

Further, I used the AdamOptimizer for optimization operation with an optimization objective of reducing the mean cross-entropy between one-hot actual labels and max softmax probability of the model with a learning rate of 0.001. The winning class is the one with the highest softmax probability during evaluation phase.

The accuracy is calculated as the fraction of correct predictions.

I am training the model over 500 EPOCHS and a BATCH_SIZE of 128.

The weights and biases are sampled from a random normal distribution witha  mean=0, stddev=0.1 
 
#### 3. Results.
(Output in the "Model generation and evaluation section of the ipynb")

**Training, validation and test:**
Here are the plots of training and validation accuracies while using the different transformations of the dataset. Each one of the datasets produced models that could achieve at least 93% accuracy over 500 epochs. The curves, as seen in the plots below (and raw values that I saw) had mostly flattened out with only minor variance in accuracy over each other Epoch - so I wasn't too hopeful of the performance to incrase with more training iterations.

**Training accuracy:**
*Training dataset:* 100% on all datasets.
*Validation dataset:* Over 93% on all datasets. Over 94% on the "Cropped images dataset", the  "Cropped_grayscale_data" and the "Cropped grayscaled normalized dataset".
*Test dataset:* *92%* for the original dataset, *92.2%* for the cropped grayscaled dataset, *92.6%* for the cropped dataset, *92.7%* for the original normalized dataset, *93.1%* for the cropped grayscaled normalized dataset, *93.7%* for the cropped normalized dataset.

All the results are very comparable and satisfactory and too statistically insignificant to declare a winner here. LeNet architecture proved to be the right choice for this problem, mainly because of the similarity of this problem to the hand-written digit recognition problem where eNet has already proven to work well. Our problem has similar characteristics where translational invariance, finding hidden features etc. will help the network classify better. I also believe that the order of the number of classes between handwritten digit recognition problem and our problem is comparable, which helps teh architecture work in our favour but I have no evidece to prove that LeNet won't perform well when the number of classes is huge.

The higher accuracy on the training dataset as compared to the validation and testing dataset points to overfitting, but since we have gotten the accuracy values we wanted, I'll move on.


## TODO: Insert plots here
**Here are the training/validation accuracy plots for all the datasets considered:**
<br/>
**Original images:**
![original_data][image16]
<br/>
**Normalized images:**
![original_normalized_data][image17]
<br/>
**Cropped images:**
![cropped_data][image18]
<br/>
**Cropped grayscaled images:**
![cropped_grayscale_data][image19]
<br/>
**Cropped normalized images:**
![cropped_normalized_data][image20]
<br/>
**Cropped grayscaled normalized images:**
![cropped_grayscaled_normalized_data][image21]
<br/>

The attached ipynb has the relevant outputs.

<br/><br/>

**Other things I would've tried if I had more time:**
* The model architecture is the same as from the LeNet lab in the course. I did not have to alter it to achieve the required acccuracy on the validation set. However, I believe that choosing a more informed (through triall and error) kernel size after analyzing the images in the dataset, considering dropout techniques, dirferent kinds of padding techniques etc. could have improved the performance and are worth exploring.
 

<br/><br/><br/>




### Testing the Model on New Images

Here are five German traffic signs that I found on the web:

![sample_test_images][image22]

The first image might be difficult to classify because it looks very similar to some other images in the training dataset (e.g. Pedestrians, General caution etc).

The second image may be difficult to classify because the training dataset doesn't have as many examples of this kind of sign. This may not matter if the training dataset has a wider variety (different angles, perspectives etc.) of this sign.

The third image looks similar to "End of all speed and passing limits" (but our model's inability for angular invariance may work in our favour) and the "No passingg sign".

The fourth image may again be misclassified due to lack of training data for it's class.

The fifth image is awfully similar to all the other speed limit signs with minor variations to show diferent digits.

#### 1. Model's performance 

**The third column of the following tables shows the top five softmax probabilities/competing predictions by the model. Here are performance figures for testing these 5 images on all the datasets considered:**
(the code for computing these results is in the 12th cell of [LeNet_eval](https://github.com/saajanis/CarND/blob/master/CarND-Project2/CarND-Traffic-Sign-Classifier-Project/LeNet_eval.ipynb) notebook that the main notebook above loads)

<br/>
**Original images:**

|       Actual value       |     Predicted value     |Other labels' probability|
|:------------------------:|:-----------------------:|:-----------------------:|
| Speed limit (70km/h)     | Children crossing       | ['Children crossing:    |
|                          |                         | 0.991454', 'Bicycles    |
|                          |                         | crossing: 0.0085176',   |
|                          |                         | 'Right-of-way at the    |
|                          |                         | next intersection:      |
|                          |                         | 2.59504e-05', 'Ahead    |
|                          |                         | only: 9.91355e-07',     |
|                          |                         | 'Speed limit (60km/h):  |
|                          |                         | 5.24917e-07']           |
| Bumpy road               | Bumpy road              | ['Bumpy road:           |
|                          |                         | 0.605363',              |
|                          |                         | 'Pedestrians:           |
|                          |                         | 0.265313', 'Road work:  |
|                          |                         | 0.122404', 'Beware of   |
|                          |                         | ice/snow: 0.00683382',  |
|                          |                         | 'General caution:       |
|                          |                         | 8.51559e-05']           |
| No entry                 | No entry                | ['No entry: 0.999994',  |
|                          |                         | 'Stop: 5.92344e-06',    |
|                          |                         | 'Road work:             |
|                          |                         | 2.57312e-17', 'Road     |
|                          |                         | narrows on the right:   |
|                          |                         | 8.04627e-18', 'Keep     |
|                          |                         | right: 1.6528e-19']     |
| Right-of-way at the next | Right-of-way at the     | ['Right-of-way at the   |
| intersection             | next intersection       | next intersection:      |
|                          |                         | 0.729084', 'Beware of   |
|                          |                         | ice/snow: 0.186675',    |
|                          |                         | 'Double curve:          |
|                          |                         | 0.0813368', 'Bicycles   |
|                          |                         | crossing: 0.00113828',  |
|                          |                         | 'Slippery road:         |
|                          |                         | 0.000969548']           |
| Stop                     | Stop                    | ['Stop: 1.0', 'No       |
|                          |                         | entry: 5.01565e-07',    |
|                          |                         | 'Road work:             |
|                          |                         | 1.53434e-09', 'Slippery |
|                          |                         | road: 8.33585e-10',     |
|                          |                         | 'Bumpy road:            |
|                          |                         | 1.75004e-11']           |

*The model trained on this dataset seems to have performed better with the 5 images (80% accuracy) than the test datset (68.4%). What's peculiar is that the model is fairly certain about it's guess that the first prediction is a "Children crossing" even though the sign looks nothing close to the "Speed limit (70km/h)" sign. It's hard to tell without visualizing the weights to find out what the neural network is seeing as similar.*

<br/>
**Normalized images:**

|       Actual value       |     Predicted value     |Other labels' probability|
|:------------------------:|:-----------------------:|:-----------------------:|
| Speed limit (70km/h)     | Speed limit (60km/h)    | ['Speed limit (60km/h): |
|                          |                         | 0.476656', 'Bicycles    |
|                          |                         | crossing: 0.166554',    |
|                          |                         | 'Speed limit (20km/h):  |
|                          |                         | 0.131353', 'Ahead only: |
|                          |                         | 0.0774043', 'General    |
|                          |                         | caution: 0.0594243']    |
| Bumpy road               | Bicycles crossing       | ['Bicycles crossing:    |
|                          |                         | 0.753388', 'Bumpy road: |
|                          |                         | 0.118509', 'No passing: |
|                          |                         | 0.111833', 'Road work:  |
|                          |                         | 0.0133663', 'Slippery   |
|                          |                         | road: 0.00249471']      |
| No entry                 | No entry                | ['No entry: 0.861382',  |
|                          |                         | 'Stop: 0.137558',       |
|                          |                         | 'Speed limit (30km/h):  |
|                          |                         | 0.000849954', 'Speed    |
|                          |                         | limit (70km/h):         |
|                          |                         | 0.000112047', 'Speed    |
|                          |                         | limit (50km/h):         |
|                          |                         | 4.41257e-05']           |
| Right-of-way at the next | Right-of-way at the     | ['Right-of-way at the   |
| intersection             | next intersection       | next intersection:      |
|                          |                         | 0.995057',              |
|                          |                         | 'Pedestrians:           |
|                          |                         | 0.00248211', 'Beware of |
|                          |                         | ice/snow: 0.00239722',  |
|                          |                         | 'Road narrows on the    |
|                          |                         | right: 2.98213e-05',    |
|                          |                         | 'Slippery road:         |
|                          |                         | 1.43127e-05']           |
| Stop                     | Stop                    | ['Stop: 0.942697', 'No  |
|                          |                         | entry: 0.0445758',      |
|                          |                         | 'Speed limit (30km/h):  |
|                          |                         | 0.00818257', 'Traffic   |
|                          |                         | signals: 0.00257886',   |
|                          |                         | 'Speed limit (20km/h):  |
|                          |                         | 0.000698614']           |

*The model trained on this dataset has a fair performance on the 5 images (60% accuracy) as compared to the test datset (82% accuracy). The first two incorrect predictions are probably because of the visual similarity between the signs.*

<br/>
**Cropped images:**

|       Actual value       |     Predicted value     |Other labels' probability|
|:------------------------:|:-----------------------:|:-----------------------:|
| Speed limit (70km/h)     | Speed limit (70km/h)    | ['Speed limit (70km/h): |
|                          |                         | 0.8145', 'Speed limit   |
|                          |                         | (30km/h): 0.141042',    |
|                          |                         | 'Speed limit (120km/h): |
|                          |                         | 0.0266006', 'Speed      |
|                          |                         | limit (20km/h):         |
|                          |                         | 0.00719198', 'Speed     |
|                          |                         | limit (100km/h):        |
|                          |                         | 0.00463047']            |
| Bumpy road               | General caution         | ['General caution:      |
|                          |                         | 0.931857',              |
|                          |                         | 'Pedestrians:           |
|                          |                         | 0.0680842', 'Bumpy      |
|                          |                         | road: 5.38497e-05',     |
|                          |                         | 'Children crossing:     |
|                          |                         | 2.18421e-06', 'Beware   |
|                          |                         | of ice/snow:            |
|                          |                         | 1.24519e-06']           |
| No entry                 | No entry                | ['No entry: 0.99064',   |
|                          |                         | 'Road work:             |
|                          |                         | 0.00900422', 'Stop:     |
|                          |                         | 0.000355437', 'Yield:   |
|                          |                         | 1.3008e-08', 'Bumpy     |
|                          |                         | road: 6.60499e-10']     |
| Right-of-way at the next | Right-of-way at the     | ['Right-of-way at the   |
| intersection             | next intersection       | next intersection:      |
|                          |                         | 0.9999', 'Slippery      |
|                          |                         | road: 6.08028e-05',     |
|                          |                         | 'Road narrows on the    |
|                          |                         | right: 2.2381e-05',     |
|                          |                         | 'General caution:       |
|                          |                         | 1.2124e-05',            |
|                          |                         | 'Pedestrians:           |
|                          |                         | 3.76229e-06']           |
| Stop                     | Priority road           | ['Priority road:        |
|                          |                         | 0.690709', 'Stop:       |
|                          |                         | 0.172003', 'Yield:      |
|                          |                         | 0.137167', 'No entry:   |
|                          |                         | 0.000120679', 'Road     |
|                          |                         | work: 4.42161e-07']     |

*The model trained on this dataset has a fair performance on the 5 images (60% accuracy) as compared to the test datset (61.3% accuracy). Since the model didn't perform well on the test dataset in the first place, I wasn't expecting it to perform well on the 5 images anyway.*

<br/>
**Cropped normalized images:**

|       Actual value       |     Predicted value     |Other labels' probability|
|:------------------------:|:-----------------------:|:-----------------------:|
| Speed limit (70km/h)     | Road work               | ['Road work: 0.470161', |
|                          |                         | 'Double curve:          |
|                          |                         | 0.266656', 'Bicycles    |
|                          |                         | crossing: 0.196566',    |
|                          |                         | 'Speed limit (60km/h):  |
|                          |                         | 0.0356204', 'Right-of-  |
|                          |                         | way at the next         |
|                          |                         | intersection:           |
|                          |                         | 0.0126688']             |
| Bumpy road               | Bumpy road              | ['Bumpy road:           |
|                          |                         | 0.920835', 'Bicycles    |
|                          |                         | crossing: 0.0523688',   |
|                          |                         | 'Road narrows on the    |
|                          |                         | right: 0.0118008',      |
|                          |                         | 'Speed limit (60km/h):  |
|                          |                         | 0.00632567', 'Dangerous |
|                          |                         | curve to the right:     |
|                          |                         | 0.00272292']            |
| No entry                 | No entry                | ['No entry: 0.993232',  |
|                          |                         | 'No vehicles:           |
|                          |                         | 0.00539223', 'Stop:     |
|                          |                         | 0.000982277', 'Speed    |
|                          |                         | limit (60km/h):         |
|                          |                         | 0.000297539', 'Speed    |
|                          |                         | limit (80km/h):         |
|                          |                         | 3.17335e-05']           |
| Right-of-way at the next | Right-of-way at the     | ['Right-of-way at the   |
| intersection             | next intersection       | next intersection:      |
|                          |                         | 0.99617', 'Beware of    |
|                          |                         | ice/snow: 0.00352904',  |
|                          |                         | 'Pedestrians:           |
|                          |                         | 0.000195199', 'Double   |
|                          |                         | curve: 8.49213e-05',    |
|                          |                         | 'General caution:       |
|                          |                         | 1.17747e-05']           |
| Stop                     | Stop                    | ['Stop: 0.924056',      |
|                          |                         | 'Speed limit (80km/h):  |
|                          |                         | 0.0246544', 'No entry:  |
|                          |                         | 0.0132418', 'Speed      |
|                          |                         | limit (70km/h):         |
|                          |                         | 0.0103568', 'Speed      |
|                          |                         | limit (30km/h):         |
|                          |                         | 0.00941374']            |

*The model trained on this dataset has a fair performance on the 5 images (80% accuracy) as compared to the test datset (85.6% accuracy), which are fairly similar. The model seems to be uncertain about the one incorrect prediction.*

<br/>

**Cropped grayscaled images:**

|       Actual value       |     Predicted value     |Other labels' probability|
|:------------------------:|:-----------------------:|:-----------------------:|
| Speed limit (70km/h)     | Road work               | ['Road work: 0.599009', |
|                          |                         | 'Speed limit (60km/h):  |
|                          |                         | 0.275431', 'Speed limit |
|                          |                         | (20km/h): 0.102531',    |
|                          |                         | 'Stop: 0.00599548',     |
|                          |                         | 'General caution:       |
|                          |                         | 0.00593451']            |
| Bumpy road               | Bumpy road              | ['Bumpy road:           |
|                          |                         | 0.848254', 'Dangerous   |
|                          |                         | curve to the left:      |
|                          |                         | 0.150462', 'Road work:  |
|                          |                         | 0.000768471', 'Yield:   |
|                          |                         | 0.000264236', 'Turn     |
|                          |                         | right ahead:            |
|                          |                         | 0.000127996']           |
| No entry                 | Stop                    | ['Stop: 0.850136',      |
|                          |                         | 'Ahead only: 0.104028', |
|                          |                         | 'No entry: 0.0448127',  |
|                          |                         | 'Speed limit (60km/h):  |
|                          |                         | 0.000418495', 'Road     |
|                          |                         | work: 0.000393552']     |
| Right-of-way at the next | Right-of-way at the     | ['Right-of-way at the   |
| intersection             | next intersection       | next intersection:      |
|                          |                         | 0.999038', 'Slippery    |
|                          |                         | road: 0.000962534',     |
|                          |                         | 'Double curve:          |
|                          |                         | 3.35115e-08', 'Children |
|                          |                         | crossing: 4.61828e-09', |
|                          |                         | 'Wild animals crossing: |
|                          |                         | 1.76708e-10']           |
| Stop                     | Keep right              | ['Keep right:           |
|                          |                         | 0.688868', 'Stop:       |
|                          |                         | 0.308289', 'Speed limit |
|                          |                         | (60km/h): 0.00143759',  |
|                          |                         | 'Speed limit (50km/h):  |
|                          |                         | 0.00126262', 'Yield:    |
|                          |                         | 7.55866e-05']           |

*The model trained on this dataset has a dismal performance on the 5 images (40% accuracy) as compared to the test datset (75.5% accuracy). THe model also seems to be fairly uncertain of it's predictions in the incorrect classifications.*

<br/>
**Cropped grayscaled normalized images:**

|       Actual value       |     Predicted value     |Other labels' probability|
|:------------------------:|:-----------------------:|:-----------------------:|
| Speed limit (70km/h)     | Road narrows on the     | ['Road narrows on the   |
|                          | right                   | right: 0.262215', 'Wild |
|                          |                         | animals crossing:       |
|                          |                         | 0.188653', 'Road work:  |
|                          |                         | 0.154681', 'Double      |
|                          |                         | curve: 0.128569',       |
|                          |                         | 'Beware of ice/snow:    |
|                          |                         | 0.0676538']             |
| Bumpy road               | Bumpy road              | ['Bumpy road:           |
|                          |                         | 0.662885', 'Keep right: |
|                          |                         | 0.183261', 'Road work:  |
|                          |                         | 0.11683', 'Go straight  |
|                          |                         | or right: 0.0283313',   |
|                          |                         | 'Bicycles crossing:     |
|                          |                         | 0.00201516']            |
| No entry                 | No entry                | ['No entry: 0.999656',  |
|                          |                         | 'Stop: 0.000230271',    |
|                          |                         | 'Turn right ahead:      |
|                          |                         | 8.31293e-05', 'Yield:   |
|                          |                         | 1.87448e-05', 'Turn     |
|                          |                         | left ahead:             |
|                          |                         | 6.11012e-06']           |
| Right-of-way at the next | Right-of-way at the     | ['Right-of-way at the   |
| intersection             | next intersection       | next intersection:      |
|                          |                         | 0.980268', 'Beware of   |
|                          |                         | ice/snow: 0.0132466',   |
|                          |                         | 'Pedestrians:           |
|                          |                         | 0.00488658',            |
|                          |                         | 'Roundabout mandatory:  |
|                          |                         | 0.000598375', 'Slippery |
|                          |                         | road: 0.00026404']      |
| Stop                     | Stop                    | ['Stop: 0.352594',      |
|                          |                         | 'Road work: 0.312202',  |
|                          |                         | 'Yield: 0.155299',      |
|                          |                         | 'Turn right ahead:      |
|                          |                         | 0.0809904', 'Speed      |
|                          |                         | limit (50km/h):         |
|                          |                         | 0.0248459']             |

*The model trained on this dataset seems to have performed better with the 5 images (80% accuracy) than the test datset (77.8%), but it is fairly comparable. It is a decent performance where the model is fairly certain about the correct predictions with an exception of the "Stop" sign's prediction where a competing sign came really close in probability.*

<br/>

#### 2. Conclusion:

3 of the 6 models tested managed to achieve 80% accuracy on the real-world dataset which is pretty impressive. The models seemed to have a hard time classifying the "Speed limit (70km/h)" sign. I can only theorize that it could be that the sign looks similar to the incorrect predictions in some layer(s) of the neural network or it may be that the input image itself has some characteristics (variance in contrast etc.) that makes it difficult to classify.
