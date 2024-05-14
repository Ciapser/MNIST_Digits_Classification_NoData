import Config
from NeuroUtils import Core

import numpy as np
import os
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.feature import hog
from sklearn.mixture import GaussianMixture
#1
#Creating Class of the project, putting parameters from Config file
Mnist = Core.Project.Classification_Project(Config)

#2
#Initializating data from main database folder to project folder. 
#Parameters of this data like resolution and crop ratio are set in Config
Mnist.Initialize_data()
####################################################



####################################################
#3
#Loading and merging data to trainable dataset.
Mnist.X_TRAIN = np.load(os.path.join(Mnist.DATA_DIRECTORY , "x_train.npy"))
Mnist.X_TEST = np.load(os.path.join(Mnist.DATA_DIRECTORY , "x_test.npy"))
Mnist.Y_TRAIN = np.load(os.path.join(Mnist.DATA_DIRECTORY , "y_train.npy"))
Mnist.Y_TEST = np.load(os.path.join(Mnist.DATA_DIRECTORY , "y_test.npy"),allow_pickle=True)

Mnist.N_CLASSES = len(Mnist.Y_TRAIN[0])
Mnist.DICTIONARY = [(element , str(element))for element in np.arange(Mnist.N_CLASSES)]


model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape  = (32,32,3))
model = tf.keras.Model(inputs=model.input, outputs=model.layers[-36].output)


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

x_train = Mnist.X_TRAIN
y_train = Mnist.Y_TRAIN
y_train = np.argmax(y_train, axis=1)
x_train1 = []
for i, image in enumerate(x_train):
    # Resize the image
    resized_image = cv2.resize(image, (32,32))
    # Convert the grayscale image to a 3-channel image
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
    # Store the resized image in the resized_images_array
    x_train1.append(resized_image)
   
x_train1 = np.array(x_train1)
x_train1 = x_train1.astype("float32")/255


a = model.predict(x_train1)
a1 = np.reshape(a,(42000,2048))
from sklearn.decomposition import PCA

pca = PCA(n_components=10)

# Fit PCA to your data and transform it
a2 = pca.fit_transform(a1)


kmeans = KMeans(n_clusters=10)
kmeans.fit(a2)




clusters = kmeans.labels_





def plot(label,cluster_assignments):
    x = x_train[cluster_assignments==label]
    for i in range(10):
        for k in range(10):
            plt.subplot(10,10,i*10+k+1)
            plt.imshow(x[i*10+k])

    


def accuracy_score(y_true, y_pred):
    # Calculate the number of correct predictions
    num_correct = np.sum(y_true == y_pred)
    
    # Calculate the total number of predictions
    total_predictions = len(y_true)
    
    # Calculate the accuracy as a percentage
    accuracy = (num_correct / total_predictions) * 100
    
    return accuracy

def map_labels(labels, label_map):
    # Define a mapping function
    def map_func(label):
        return label_map.get(label, label)  # If label not found in dictionary, return original label
    
    # Vectorize the mapping function
    vectorized_map_func = np.vectorize(map_func)
    
    # Map labels using the vectorized function
    new_labels = vectorized_map_func(labels)
    return new_labels

plot(0,clusters)




# Initialize the best label mapping and its accuracy score
best_label_map = {i: i for i in range(10)}  # Initialize with identity mapping
best_accuracy = accuracy_score(y_train, map_labels(clusters, best_label_map))

# Iterate through each cluster label
for label in range(10):  # Skip label 0 since it's already initialized as itself
    # Randomly sample label mappings for the current label while keeping other mappings fixed
    current_label_map = best_label_map.copy()
    label_candidates = list(range(10))
    label_candidates.remove(label)  # Exclude the current label
    np.random.shuffle(label_candidates)
    for candidate_label in label_candidates:
        current_label_map[label] = candidate_label
        # Evaluate accuracy with the current label mapping
        current_accuracy = accuracy_score(y_train, map_labels(clusters, current_label_map))
        # Update the best label mapping if the accuracy improves
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_label_map = current_label_map.copy()

print("Best Label Mapping:", best_label_map)
print("Best Accuracy Score:", best_accuracy)

        
y_pred = map_labels(clusters, best_label_map)
accuracy_score(y_train, y_pred)


"""
#8
#Making submission
sample_submission = Mnist.Generate_sample_submission()
sample_submission.to_csv(os.path.join(Mnist.MODEL_DIRECTORY,"sample_submission.csv"), index = False)
"""











