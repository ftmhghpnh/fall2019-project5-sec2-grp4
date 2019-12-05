# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 20:16:15 2019

@author: tb2579
"""
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pprint
pp = pprint.PrettyPrinter(indent=4)
import cv2
import pandas
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale

 
# Function to extract features from images
def ExtractFeatures(image1):
    image1=rescale(image1, 2/3, mode='reflect')
    image1_hog, image1_hog_img = hog(image1,
                     pixels_per_cell=(12, 12),
                     cells_per_block=(2,2),
                     orientations=8,
                     visualise=True,
                     block_norm='L2-Hys')
    return image1_hog


# Reading file containing image locations
df = pandas.read_csv('C:/Users/tb2579/Documents/MA Statistics/Applied Data Analysis/Project 5/fileNames.csv')

trainIndex=78241
testIndex=19559

features_df= pandas.DataFrame()

for img_file in range(trainIndex, 97800):
    image1 = imread(df["Image"][img_file], as_grey=True)
    image1_hog=ExtractFeatures(image1)
    print(image1_hog)
    
     
    image1_hog=pandas.Series(image1_hog)
    features_df = features_df.append(image1_hog, ignore_index=True)
 
features_df.to_csv("C:/Users/tb2579/Documents/MA Statistics/Applied Data Analysis/Project 5/TrainFeatures_97800.csv")

