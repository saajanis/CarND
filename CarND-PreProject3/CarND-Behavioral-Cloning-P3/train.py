import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle

#################
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
#################

# IMG_DIR = './data/training/IMG/'
#  
# lines = []
# data = {'front_images': [], 'left_images': [], 'right_images': [], 
#         'steering_angles_front': [], 'steering_angles_left': [], 'steering_angles_right': []}
#  
# with open('./data/training/driving_log.csv') as csvFile:
#     reader = csv.reader(csvFile)
#     for line in reader:
#         lines.append(line)   
#  
# lines_pbar = tqdm(lines, unit='lines')
# for line in lines_pbar:
#     front_filename = line[0].split('/')[-1]
#     left_filename = line[1].split('/')[-1]
#     right_filename = line[2].split('/')[-1]
#     
#     #TODO: Tune this parameter
#     steering_correction = 0.2
#      
#     #TODO: Create mask for the road (bottom 50%?) 
#     # With augmentation (flipped and steering_angle reversed)
#     front_image = cv2.imread(IMG_DIR + front_filename)
#     steering_angle = np.float(line[3])
#     data['front_images'].append(front_image)
#     data['front_images'].append(cv2.flip(front_image, 1))
#     data['steering_angles_front'].append(steering_angle)
#     data['steering_angles_front'].append(-1.0 * steering_angle)
#     
#     left_image = cv2.imread(IMG_DIR + left_filename)
#     steering_angle_left = np.float(line[3]) + steering_correction
#     data['left_images'].append(left_image)
#     data['left_images'].append(cv2.flip(left_image, 1))
#     data['steering_angles_left'].append(steering_angle_left)
#     data['steering_angles_left'].append(-1.0 * steering_angle_left)
#     
#     right_image = cv2.imread(IMG_DIR + right_filename)
#     steering_angle_right = np.float(line[3]) - steering_correction
#     data['right_images'].append(right_image)
#     data['right_images'].append(cv2.flip(right_image, 1))
#     data['steering_angles_right'].append(steering_angle_right)
#     data['steering_angles_right'].append(-1.0 * steering_angle_right)
#     
#  
# pickle_dump(data, "data.p")
data = pickle_load("data.p")

print('Data creation/loading completed...')

# TODO: Can't think of a way why testing data can be beneficial (cheating by testing in simulator anyways)
# X_train, X_test, y_train, y_test = train_test_split(
#     data['front_images'] + data['left_images'] + data['right_images'], 
#     data['steering_angles_front'] + data['steering_angles_left'] + data['steering_angles_right'], 
#     test_size=0.33, random_state = 42)
  
X_train = data['front_images'] + data['left_images'] + data['right_images']
y_train = data['steering_angles_front'] + data['steering_angles_left'] + data['steering_angles_right']
    
# Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Normalized data with zero mean
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

# TODO: Use all 3 images
model.compile(optimizer='adam', loss='mse')
model.fit(np.array(X_train), np.array(y_train), validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
    