import Config
from NeuroUtils import Core
"""
#For working with unreleased version directly from local repository, to test things out
import sys
sys.path.insert(0, "Path_to_local_library_repository")
"""

#1
#Creating Class of the project, putting parameters from Config file
Cifar10 = Core.Project.Classification_Project(Config)

#2
#Initializating data from main database folder to project folder. 
#Parameters of this data like resolution and crop ratio are set in Config
Cifar10.Initialize_data()

#3
#Loading and merging data to trainable dataset.
#Optional reduction of the size class
Cifar10.Load_and_merge_data()

#4
#Processing data by splitting it to train,val and test set and data augmentation.
#Parameters of augmentation and reduction are set in the Config file
Cifar10.Process_data()

#5
#Initialization of model architecture from library. 
#Model architecture is specified in the config
#This step can be skipper and you can provide your own compiled model by:
"""    
Cifar10.MODEL = custom_model    
"""
Cifar10.Initialize_model_from_library()

#6
#Training of the model. It can load previously saved data from project folder
#or train from scratch. Parameters of training are set in Config file
Cifar10.Initialize_weights_and_training()

#7
#Showing results of the training and evaluates the model
#Parameters of results can be set in Config file
Cifar10.Initialize_resulits()




