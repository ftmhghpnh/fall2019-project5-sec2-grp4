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

 # Training - Using SGD (Stochastic Gradient Descent)
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

train_df=pandas.read_csv('C:/Users/tb2579/Documents/MA Statistics/Applied Data Analysis/Project 5/TrainFeatures.csv')
test_df=pandas.read_csv('C:/Users/tb2579/Documents/MA Statistics/Applied Data Analysis/Project 5/TestFeatures.csv')

train_features=train_df.iloc[:,0:1152]
y_train=train_df["Label"]


sgd_clf = SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
sgd_clf.fit(train_features, y_train)

# Testing
test_features=test_df.iloc[:,0:1152]
y_test=test_df["Label"]

y_pred = sgd_clf.predict(test_features)
print(np.array(y_pred == y_test)[:25])
print('')
print('Percentage correct: ', 100*np.sum(y_pred == y_test)/len(y_test))


df = pandas.DataFrame(
    np.c_[y_test, y_pred],
    columns=['true_label', 'prediction']
)
df.head(20)


