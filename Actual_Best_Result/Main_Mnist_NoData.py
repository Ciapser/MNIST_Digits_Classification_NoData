import Config
from NeuroUtils import Core

import numpy as np
import os
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#1
#Creating Class of the project, putting parameters from Config file
Mnist = Core.Project.Classification_Project(Config)

#2
#Initializating data from main database folder to project folder. 
#Its divided to train and test by kaggle so its loaded from the provided csv
Mnist.Initialize_data()
####################################################



####################################################
#3
#Loading and merging data to trainable dataset. Hovewer it wont be used to train data, its only for evaluation
x_train = np.load(os.path.join(Mnist.DATA_DIRECTORY , "x_train.npy"))
x_test = np.load(os.path.join(Mnist.DATA_DIRECTORY , "x_test.npy"))
y_train = np.load(os.path.join(Mnist.DATA_DIRECTORY , "y_train.npy"))
y_train = np.argmax(y_train, axis=1)
y_test = np.load(os.path.join(Mnist.DATA_DIRECTORY , "y_test.npy"),allow_pickle=True)

n_classes = len(y_train)
dictionary = [(element , str(element))for element in np.arange(n_classes)]

# Preparing model
model = tf.keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape  = (32,32,3))
model = tf.keras.Model(inputs=model.input, outputs=model.layers[-36].output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
#Loading data into the





#Plot function which shows 100 images of given class
def plot(data,class_to_plot,labels):
    x = data[labels==class_to_plot]
    for i in range(10):
        for k in range(10):
            plt.subplot(10,10,i*10+k+1)
            plt.imshow(x[i*10+k])

#Plots accuracy score            
def accuracy_score(y_true, y_pred):
    # Calculate the number of correct predictions
    num_correct = np.sum(y_true == y_pred)
    
    # Calculate the total number of predictions
    total_predictions = len(y_true)
    
    # Calculate the accuracy as a percentage
    accuracy = (num_correct / total_predictions) * 100
    
    return accuracy

#Mapping labels to different labels provided in label map
def map_labels(labels, label_map):
    # Define a mapping function
    def map_func(label):
        return label_map.get(label, label)  # If label not found in dictionary, return original label
    
    # Vectorize the mapping function
    vectorized_map_func = np.vectorize(map_func)
    
    # Map labels using the vectorized function
    new_labels = vectorized_map_func(labels)
    return new_labels

#Predicting classes based on pretrained model, pca and k means. No training is involved
def Predict_No_FineTuning(x,pretrained_model):
    x_resized = []
    for i, image in enumerate(x):
        # Resize the image
        resized_image = cv2.resize(image, (32,32))
        # Convert the grayscale image to a 3-channel image
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
        # Store the resized image in the resized_images_array
        x_resized.append(resized_image)
       
    x_resized  = np.array(x_resized)
    x_resized = x_resized.astype("float32")/255


    a = pretrained_model.predict(x_resized)
    resize_shape = int(a.shape[1])*int(a.shape[2])*int(a.shape[3])
    a = np.reshape(a,(len(a),resize_shape))

    # Fit PCA to your data and transform it
    pca = PCA(n_components=10)
    a = pca.fit_transform(a)

    #k_means clustering
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(a)
    clusters = kmeans.labels_
    try:
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
                    
        predicted = map_labels(clusters, best_label_map)
        return predicted
    except:
        predicted = clusters
        return predicted




#Predicting values on train
predicted= Predict_No_FineTuning(x_train,model)
print("Accuracy score: ",accuracy_score(y_train, predicted))
#plot(5,predicted)

#Predicting values on test
predicted = Predict_No_FineTuning(x_test,model)
best_label_map = {0:6,
                  1:7,
                  2:1,
                  3:9,
                  4:0,
                  5:3,
                  6:2,
                  7:1,
                  8:5,
                  9:4,           
    }


predicted = map_labels(predicted, best_label_map)
#plot(x_test,5,predicted)

#Making submission
Mnist.Y_TEST = predicted
sample_submission = Mnist.Generate_sample_submission()
sample_submission.to_csv("sample_submission.csv", index = False)












