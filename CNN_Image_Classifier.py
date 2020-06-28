import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
from dataGenerator import data_generator
from PIL import Image
from tqdm import tqdm
import time
import glob
import random
import pickle
import cv2
import os

#Data Augmentation
data_generator()

#load gray images
data = '/home/mzwandile/Downloads/CNN/data/'
wild_cats = ["wildcats_che", "wildcats_leo"]

#Image Visualization
for cat in wild_cats:
    path = os.path.join(data, cat)
    for cat_image in os.listdir(path):
        image_list = cv2.imread(os.path.join(path, cat_image), cv2.IMREAD_GRAYSCALE)
        #plt.figure(figsize = (5, 4))
        #plt.imshow(image_list, cmap = 'gray')
        #plt.show()
        break
    break  
 

#specify required image size for the model [to make it uniform (n x n)]
image_size = 50 #this you can change

new_array = cv2.resize(image_list, (image_size, image_size))
#plt.figure(figsize = (5, 4))
#plt.imshow(new_array, cmap='gray')
#plt.show()

training_data = []

def make_featutes():
    for cat in wild_cats:  
        path = os.path.join(data, cat) 
        class_num = wild_cats.index(cat)  # create classes  '0' for cheetah and '1' for leopard
        for image in tqdm(os.listdir(path)):  # iterate over each image for each cat (leopard and cheetah)
            try:
                image_array = cv2.imread(os.path.join(path, image) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(image_array, (image_size, image_size))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add to training_data
            except Exception as e:  
                pass

make_featutes()

print(len(training_data))

#shuffle the images
random.shuffle(training_data)

for sample in training_data[:6]:
    print(sample[1])


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X)

pickle_out = open('X_pickle', 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open('y_pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()



pickle_in = open('X_pickle', 'rb')
X = pickle.load(pickle_in)

pickle_in = open('y_pickle', 'rb')
y = pickle.load(pickle_in)

print(X[1])
'''
plt.figure(figsize = (5, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X[i], cmap = 'gray')
    plt.ylabel(y[i])
    plt.xticks([])
    plt.yticks([])

plt.savefig('wildcat_grid.eps')
plt.show()
'''

X = X/255.0

#Split data into training and testing sets
train_features, test_features, train_targets, test_targets = train_test_split(X, y, random_state = 42, test_size = 0.25)

train_features = np.array(train_features)
test_features = np.array(test_features)
train_targets = np.array(train_targets)
test_targets = np.array(test_targets)

train_features = train_features.reshape(len(train_features), image_size, image_size, 1)
test_features = test_features.reshape(len(test_features), image_size, image_size, 1)


#Initialize a CNN model
filter_size = (3, 3)
pool_size = (2, 2)
cnn_model = keras.models.Sequential([
            keras.layers.Conv2D(32, filter_size, activation = 'relu', input_shape = (image_size, image_size, 1)),
            keras.layers.MaxPooling2D(pool_size = pool_size),
            keras.layers.Dropout(0.5),
            
            keras.layers.Conv2D(64, filter_size, activation = 'relu', padding = 'same'),
            keras.layers.MaxPooling2D(pool_size = pool_size),
            keras.layers.Dropout(0.5),

            keras.layers.Conv2D(64, filter_size, activation = 'relu', padding = 'same'),
            keras.layers.MaxPooling2D(pool_size = pool_size),
            keras.layers.Dropout(0.5),

            keras.layers.Flatten(),
            keras.layers.Dense(250, activation = 'relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation = 'sigmoid'),
            
    ]
)


#Compiling
cnn_model.compile(loss = 'binary_crossentropy',
                  optimizer = keras.optimizers.Adam(lr = 1e-3),
                  metrics = ['accuracy']
)

print(cnn_model.summary())

# fitting (training)
training_history = cnn_model.fit(train_features, train_targets, validation_split = 0.33, batch_size = None, epochs = 130, verbose = True)

# Get training history
hist = pd.DataFrame(training_history.history)
iterations = np.array(np.arange(1, 1 + len(hist)))
hist.insert(4, 'epoch', iterations, True)
print('')
print(hist.tail(60))

# Visualize history for accuracy
plt.figure(figsize = (5, 4))
plt.plot(hist.epoch, hist.accuracy, color = 'red', label = 'training')
plt.plot(hist.epoch, hist.val_accuracy, color = 'black', label = 'validation')
plt.ylabel('Accuracy', weight = 'bold')
plt.xlabel('Iterations', weight = 'bold')
plt.legend(loc = 'best')
plt.savefig('CNN_Model_Acuracy.eps')
plt.show()

# Visualize history for loss
plt.figure(figsize = (5, 4))
plt.plot(hist.epoch, hist.loss, color = 'red', label = 'training')
plt.plot(hist.epoch, hist.val_loss, color = 'black', label = 'validation')
plt.ylabel('Loss', weight = 'bold')
plt.xlabel('Iterations', weight = 'bold')
plt.legend(loc = 'best')
plt.savefig('CNN_Model_Loss.eps')
plt.show()


# Evaluate model on the dataset reserved for testing
score = cnn_model.evaluate(test_features, test_targets, verbose = 0)
print('')
print('Accuracy: ', round(score[1]*100, 2))

# Make predictions
y_pred_cnn = cnn_model.predict(test_features)

probability = pd.DataFrame({
    'Probability' : y_pred_cnn.flatten()
    }
)

#Convert probabilities to classes in binary form
for i in range(len(y_pred_cnn)):
    if y_pred_cnn[i] < 0.5:
        y_pred_cnn[i] = 0
    else:
        y_pred_cnn[i] = 1


#get animal names
name = []
for animal in range(len(y_pred_cnn)):
    if y_pred_cnn[animal] == 1:
        name.append('Leopard')
    else:
        name.append('Cheetah')


#Save results in a dataframe
save_cnn = pd.DataFrame({
        "Probability": probability.Probability,
        "Expected": test_targets,
        "Prediction": y_pred_cnn.flatten(),
        "Name": name,
        "Outcome": y_pred_cnn.flatten() == test_targets
    }
)

save_cnn.to_csv("Image_predictions_CNN_wild_cats.csv", index = False)
print(save_cnn.head(50))
