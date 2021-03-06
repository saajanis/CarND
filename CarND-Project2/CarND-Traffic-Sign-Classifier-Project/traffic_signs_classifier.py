import matplotlib.pyplot as plt
# Load pickled data
import pickle
import numpy as np
from collections import defaultdict
from collections import OrderedDict
from operator import itemgetter
import collections

def build_feature_label_ranked_dict(X, Y):
    assert(len(X) == len(Y))
    label_feature_dict = {}
    for x,y in zip(X, Y):
        if y not in label_feature_dict:
            label_feature_dict[y] = x
    
    label_count_dict = defaultdict(int)
    for x,y in zip(X, Y):
        label_count_dict[y] += 1
    
    ranked_count_to_label_feature_dict = collections.OrderedDict()
    for label, count in sorted(label_count_dict.items(), key=lambda x:x[1]):
        ranked_count_to_label_feature_dict[count] = (label, label_feature_dict[label])
    
    return ranked_count_to_label_feature_dict

# TODO: Fill this in based on where you saved the training and testing data

DATA_DIR = "./data/"

training_file = DATA_DIR + "traffic-signs-data/train.p"
validation_file = DATA_DIR + "traffic-signs-data/valid.p"
testing_file = DATA_DIR + "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, X_train_coords, X_train_sizes, y_train = np.asarray(train['features'], dtype=np.float32), train['coords'], train['sizes'], train['labels']
X_valid, X_valid_coords,X_valid_sizes, y_valid = np.asarray(valid['features'], dtype=np.float32), valid['coords'], valid['sizes'], valid['labels']
X_test, X_test_coords, X_test_sizes, y_test = np.asarray(test['features'], dtype=np.float32), test['coords'], test['sizes'], test['labels']


build_feature_label_ranked_dict(X_train, y_train)

#################

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

##################


### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.
#TODO(saajan): uncomment line below and plot and do more visualization
# %matplotlib inline
 
# plt.hist(y_train, alpha=0.5, label='training_labels', bins=43)
# plt.hist(y_valid, alpha=0.5, label='validation_labels', bins=43)
# plt.hist(y_test, alpha=0.5, label='test_labels', bins=43)
# plt.legend(loc='upper right')
# plt.show()

###################
import random

def show_sample_images(Xs, count):
    fig = plt.figure()
    for i in range(count):
        index = random.randint(0, len(Xs)-1)
        image = Xs[index].squeeze()
        ax1 = fig.add_subplot(1,count,i+1)
        ax1.imshow(image)
        #ax1.imshow(image, cmap="gray")
        
        #plt.imshow(image, cmap="gray")
        #plt.imshow(image)
           
# show_sample_images(X_train, 10)

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.    

def saveAndRetrievePickle(transformFunc, featuresDict, mode, pickleFileName):
    if mode == 'saveAndRetrieve':
        X_train_features = np.array([transformFunc(X_train_image) for X_train_image in featuresDict['X_train']])
        X_valid_features = np.array([transformFunc(X_valid_image) for X_valid_image in featuresDict['X_valid']])
        X_test_features = np.array([transformFunc(X_test_image) for X_test_image in featuresDict['X_test']])
             
        new_features_dict = {'X_train': X_train_features, 'y_train': y_train, 'X_valid': X_valid_features,\
                            'y_valid': y_valid, 'X_test': X_test_features, 'y_test': y_test}
    
        with open(DATA_DIR + pickleFileName + '.pickle', 'wb') as handle:
            pickle.dump(new_features_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
      
    if mode == 'saveAndRetrieve' or mode == 'retrieve':
        with open(DATA_DIR + pickleFileName + '.pickle', 'rb') as handle:
            new_features_dict = pickle.load(handle)
    
    return new_features_dict
            
   
# Original_data
def noop_image(image):
    return np.asarray(image, dtype=np.float32) 

original_data = {'X_train': X_train, 'y_train': y_train, 'X_valid': X_valid, 'y_valid': y_valid, 'X_test': X_test, 'y_test': y_test}
# original_data = saveAndRetrievePickle(noop_image, original_data, 'saveAndRetrieve', 'original_data')
original_data = saveAndRetrievePickle(noop_image, original_data, 'retrieve', 'original_data') 


# Create cropped data
#TODO(saajan): Reconsider zeroing out irrelevant area over resizing cropped image
def extract_bounds_and_rescale(image, coord, size):
    transformed_x = 32
    transformed_y = 32
    original_x = size[0]
    original_y = size[1]
      
    x_multiplier = float(transformed_x)/float(original_x)
    y_multiplier = float(transformed_y)/float(original_y)
      
    transformed_coord = (coord[0]* x_multiplier, coord[1] * y_multiplier, coord[2] * x_multiplier, coord[3] * y_multiplier)
    transformed_coord = [int(np.rint(val)) for val in transformed_coord]
      
    ret_image = image.copy()
    shape = image.shape
     
    ret_image[0:transformed_coord[0],:] = (0,0,0)
    ret_image[:,0:transformed_coord[1]] = (0,0,0)
    ret_image[transformed_coord[2]:shape[1],:] = (0,0,0)
    ret_image[:,transformed_coord[3]:shape[0]] = (0,0,0)
    #show_sample_images([ret_image], 1)
    return np.asarray(ret_image, dtype=np.float32)


# # extract_bounds_and_rescale Xs
# X_train = np.array([extract_bounds_and_rescale(image, coord, size) for (image, coord, size) in zip(X_train, X_train_coords, X_train_sizes)])
# X_valid = np.array([extract_bounds_and_rescale(image, coord, size) for (image, coord, size) in zip(X_valid, X_valid_coords,X_valid_sizes)])
# X_test = np.array([extract_bounds_and_rescale(image, coord, size) for (image, coord, size) in zip(X_test, X_test_coords, X_test_sizes)])
#      
# cropped_data = {'X_train': X_train, 'y_train': y_train, 'X_valid': X_valid, 'y_valid': y_valid, 'X_test': X_test, 'y_test': y_test}
# with open(DATA_DIR + 'cropped_data.pickle', 'wb') as cropped_data_handle:
#     pickle.dump(cropped_data, cropped_data_handle, protocol=pickle.HIGHEST_PROTOCOL)
  
with open(DATA_DIR + 'cropped_data.pickle', 'rb') as cropped_data_handle:
    cropped_data = pickle.load(cropped_data_handle)
     
# show_sample_images(cropped_data['X_train'], 10)


# normalize Xs
def normalize(image):
    normalizer_func = np.vectorize(lambda val: np.float32((float(val)-128.0)/128.0))
    return np.asarray(normalizer_func(image), dtype=np.float32)

# cropped_normalized_data = saveAndRetrievePickle(normalize, cropped_data, 'saveAndRetrieve', 'cropped_normalized_data')
cropped_normalized_data = saveAndRetrievePickle(normalize, cropped_data, 'retrieve', 'cropped_normalized_data')
   
# Experiments
# vals = []
# for image in cropped_normalized_data['X_train']:
#     index = random.randint(0, len(cropped_normalized_data['X_train'])-1)
#     vals.append((np.mean(cropped_normalized_data['X_train'][index])))
# print(np.mean(vals))    


import cv2
# convert_to_grayscale Xs
def convert_to_grayscale(image):
    return np.asarray(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), dtype=np.float32).reshape((32, 32, 1))
    
# cropped_grayscale_data = saveAndRetrievePickle(convert_to_grayscale, cropped_data, 'saveAndRetrieve', 'cropped_grayscale_data')
cropped_grayscale_data = saveAndRetrievePickle(convert_to_grayscale, cropped_data, 'retrieve', 'cropped_grayscale_data') 

# Cropped grayscale normalized
# cropped_grayscaled_normalized_data = saveAndRetrievePickle(normalize, cropped_grayscale_data, 'saveAndRetrieve', 'cropped_grayscaled_normalized_data')
cropped_grayscaled_normalized_data = saveAndRetrievePickle(normalize, cropped_grayscale_data, 'retrieve', 'cropped_grayscaled_normalized_data') 


all_data = [original_data, cropped_data, cropped_normalized_data, cropped_grayscale_data, cropped_grayscaled_normalized_data] 

print('Done with data prep!')

'''
for data_dict in all_data:
    current_data_dict = data_dict
    X_train, y_train           = current_data_dict['X_train'], current_data_dict['y_train']
    X_validation, y_validation = current_data_dict['X_valid'], current_data_dict['y_valid']
    X_test, y_test             = current_data_dict['X_test'], current_data_dict['y_test']
    
    assert(len(X_train) == len(y_train))
    assert(len(X_validation) == len(y_validation))
    assert(len(X_test) == len(y_test))
    
    print()
    print("Image Shape: {}".format(X_train[0].shape))
    print()
    print("Training Set:   {} samples".format(len(X_train)))
    print("Validation Set: {} samples".format(len(X_validation)))
    print("Test Set:       {} samples".format(len(X_test)))
    
    
    # The MNIST data that TensorFlow pre-loads comes as 28x28x1 images.
    # 
    # However, the LeNet architecture only accepts 32x32xC images, where C is the number of color channels.
    # 
    # In order to reformat the MNIST data into a shape that LeNet will accept, we pad the data with two rows of zeros on the top and bottom, and two columns of zeros on the left and right (28+2+2 = 32).
    # 
    # You do not need to modify this section.
    
    # In[2]:
    
    
    import numpy as np
    
    # Pad images with 0s
    # X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    # X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    # X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
        
    # print("Updated Image Shape: {}".format(X_train[0].shape))
    
    
    # ## Visualize Data
    # 
    # View a sample from the dataset.
    # 
    # You do not need to modify this section.
    
    # In[3]:
    
    
    import random
    import numpy as np
    import matplotlib.pyplot as plt
#     get_ipython().magic('matplotlib inline')
    
#     index = random.randint(0, len(X_train))
#     image = X_train[index].squeeze()
#     print(image.shape)
#     
#     plt.figure(figsize=(1,1))
#     plt.imshow(image, cmap="gray")
#     print(y_train[index])
    
    
    # ## Preprocess Data
    # 
    # Shuffle the training data.
    # 
    # You do not need to modify this section.
    
    # In[4]:
    
    
    from sklearn.utils import shuffle
    
    X_train, y_train = shuffle(X_train, y_train)
    
    
    # ## Setup TensorFlow
    # The `EPOCH` and `BATCH_SIZE` values affect the training speed and model accuracy.
    # 
    # You do not need to modify this section.
    
    # In[5]:
    
    
    import tensorflow as tf
    
    EPOCHS = 10
    BATCH_SIZE = 128
    
    
    # ## TODO: Implement LeNet-5
    # Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.
    # 
    # This is the only cell you need to edit.
    # ### Input
    # The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.
    # 
    # ### Architecture
    # **Layer 1: Convolutional.** The output shape should be 28x28x6.
    # 
    # **Activation.** Your choice of activation function.
    # 
    # **Pooling.** The output shape should be 14x14x6.
    # 
    # **Layer 2: Convolutional.** The output shape should be 10x10x16.
    # 
    # **Activation.** Your choice of activation function.
    # 
    # **Pooling.** The output shape should be 5x5x16.
    # 
    # **Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.
    # 
    # **Layer 3: Fully Connected.** This should have 120 outputs.
    # 
    # **Activation.** Your choice of activation function.
    # 
    # **Layer 4: Fully Connected.** This should have 84 outputs.
    # 
    # **Activation.** Your choice of activation function.
    # 
    # **Layer 5: Fully Connected (Logits).** This should have 10 outputs.
    # 
    # ### Output
    # Return the result of the 2nd fully connected layer.
    
    # In[40]:
    
    
    from tensorflow.contrib.layers import flatten
    
    def conv2d(x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)
    
    def maxpool2d(x, k=2):
        return tf.nn.max_pool(
            x,
            ksize=[1, k, k, 1],
            strides=[1, k, k, 1],
            padding='SAME')
    
    def LeNet(x):    
        # TODO(saajan): Use this function
        # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
        mu = 0
        sigma = 0.1
        n_classes = 43
        dropout = 0.25  # Dropout, probability to keep units
        
        weights = {
        'wc1': tf.Variable(tf.random_normal([5, 5, x.shape[3].value, 6], mean=mu, stddev=sigma)),
        'wc2': tf.Variable(tf.random_normal([5, 5, 6, 16], mean=mu, stddev=sigma)),
        'wd1': tf.Variable(tf.random_normal([400, 120], mean=mu, stddev=sigma)),
        'wd2': tf.Variable(tf.random_normal([120, 84], mean=mu, stddev=sigma)),
        'out': tf.Variable(tf.random_normal([84, n_classes], mean=mu, stddev=sigma))
            
        }
    
        biases = {
            'bc1': tf.Variable(tf.random_normal([6])),
            'bc2': tf.Variable(tf.random_normal([16])),
            'bd1': tf.Variable(tf.random_normal([120])),
            'bd2': tf.Variable(tf.random_normal([84])),
            'out': tf.Variable(tf.random_normal([n_classes]))
            
        }
        
        # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
        # TODO: Activation.
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
        conv1 = maxpool2d(conv1, k=2)
        
        # TODO: Layer 2: Convolutional. Output = 10x10x16.
        # TODO: Activation and dropout.
        conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
        # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
        conv2 = maxpool2d(conv2, k=2)
    
        # TODO: Flatten. Input = 5x5x16. Output = 400.
        flat = tf.contrib.layers.flatten(conv2)
        
        # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
        fc1 = tf.add(tf.matmul(flat, weights['wd1']), biases['bd1'])
        # TODO: Activation.
        fc1 = tf.nn.relu(fc1)
    
        # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
        # TODO: Activation snd dropout.
        fc2 = tf.nn.relu(fc2)
    
        # TODO: Layer 5: Fully Connected. Input = 84. Output = 43.
        logits = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
        
        return logits
    
    
    # ## Features and Labels
    # Train LeNet to classify [MNIST](http://yann.lecun.com/exdb/mnist/) data.
    # 
    # `x` is a placeholder for a batch of input images.
    # `y` is a placeholder for a batch of output labels.
    # 
    # You do not need to modify this section.
    
    # In[36]:
    
    
    x = tf.placeholder(tf.float32, (None, 32, 32, X_train[0].shape[2]))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, 43)
    
    
    # ## Training Pipeline
    # Create a training pipeline that uses the model to classify MNIST data.
    # 
    # You do not need to modify this section.
    
    # In[37]:
    
    
    rate = 0.001
    
    logits = LeNet(x)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)
    
    
    # ## Model Evaluation
    # Evaluate how well the loss and accuracy of the model for a given dataset.
    # 
    # You do not need to modify this section.
    
    # In[38]:
    
    
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    
    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
    
    
    # ## Train the Model
    # Run the training data through the training pipeline to train the model.
    # 
    # Before each epoch, shuffle the training set.
    # 
    # After each epoch, measure the loss and accuracy of the validation set.
    # 
    # Save the model after training.
    # 
    # You do not need to modify this section.
    
    # In[39]:
    
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        
        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
                
            validation_accuracy = evaluate(X_validation, y_validation)
            print("EPOCH {} ...".format(i+1))
            print("Validation Accuracy = {:.3f}".format(validation_accuracy))
            print()
            
        saver.save(sess, './lenet')
        print("Model saved")
    
    
    # ## Evaluate the Model
    # Once you are completely satisfied with your model, evaluate the performance of the model on the test set.
    # 
    # Be sure to only do this once!
    # 
    # If you were to measure the performance of your trained model on the test set, then improve your model, and then measure the performance of your model on the test set again, that would invalidate your test results. You wouldn't get a true measure of how well your model would perform against real data.
    # 
    # You do not need to modify this section.
    
    # In[41]:
    
    
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
    
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
    
    
    # In[ ]:

'''



