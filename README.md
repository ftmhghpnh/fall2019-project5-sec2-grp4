# ADS Project 5: 

Term: Fall 2018

+ Team # Group 4
+ Projec title: Landmark Recognition Challenge

+ Team members
	+ Fateme Sadat Haghpanah (Team Leader)
	+ Thomson Batidzirai
	+ Mo Yang
	+ Yian Huang
	
+ Project summary: 

This project we tried to build some models to recognize and classify the landmark. For those who are intrested in taking pictures of everthying, it could be really helpful to recognize the landmarks they pictured and remined them their name!

We did this project doing base and advanced model. For the base model, we extractd the HOG features and then used SGD to classify them. 
For advance model, first we used a 3 layered CNN and check the test and train accuracy. Second, we used a pretrained model of Xception and extracted features, and then used a 3 layered fully connected network for classification.

You can find our presentation [here](doc/project5_group04.pdf). Also, the main file is [here](doc/Main.pdf).
	
**Contribution statement**:
+ Fateme Sadat Haghpanah (Team Leader):
	- Sampling the data set and split it to test and train set
	- Run two differnt advance model (CNN-based) 
	- Make the main.Rmd and clean the github
	- Create the team and lead the project
	- Prepare the slides and do the presentation
+ Thomson Batidzirai
	- Download the data set from Kaggle 
	- Work on the base model ( extracting features and run the classification model)
+ Mo Yang
	- Extract the features for base model (HOG features)
	- edit some of the README.md files

+ Yian Huang
	- descriptive analysis of 4132914 train data
 

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
