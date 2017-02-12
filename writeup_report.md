#**Behavioral Cloning** 

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model_architecture]: ./images/model.png "Model Architechture"
[center_example]: ./images/center-example.png "Center example"
[left_recovery]: ./images/left-recovery.png "Left Recovery"
[right_recovery]: ./images/right-recovery.png "Right Recovery"
[cropped_and_flipped]: ./images/cropped-and-flipped.png "Cropped and flipped"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed (lines 28-57)

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 30). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 44, 47, and 50). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (line 72). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 55).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and removal of zero steering bias. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to read white papers and talk to people to find existing models that have been created to solve similar problems. 

At first I tried to use the [comma.ai model](https://github.com/commaai/research), but I ended up going with the [nvidia model](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) after some experimentation of training my model with small sample datasets. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I introduced dropout layers.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, but this was because I only trained the model with a small subset of images and a few ephochs. I experimented with increasing the amount of images I trained with and number of epochs until the vehicle was able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

Here is a visualization of the final model architecture (model.py lines 29-55):

![alt text][model_architecture]

####3. Creation of the Training Set & Training Process

First, I checked out the center images to get an idea what I was working with:

![alt text][center_example]

Then I explored the left and right images with calculated recovery angels:

![alt text][left_recovery]
![alt text][right_recovery]

To augment the dataset, I also flipped images and angles to try and help the model learn more about different types of turns. I also cropped the image to git rid of most of the sky since I only wanted to train my model with features on the road. Here is an image that has been cropped and flipped:

![alt text][cropped_and_flipped]

I used [sample training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by Udacity that had 8036 data points. I then preprocessed this data by removing 95% of images with 0 steering angle to remove bias for going straight, augmenting with recovery angels and flipping, and cropping the images. After the preprocessing, I ended up with 24672 data points. I only ended up training my model with a random sample of only 2000 data points though. The car was able to successfully make it around the track continously even though I only trained my model with that small dataset. When I have more time, I plan to try training with the full set and to adjust the architecture to try and reduce the number of parameters needed.

I used Keras to train the model, using 20% of the data for validation. Keras shuffled the data for me. The validation set helped determine if the model was over or under fitting. I found the ideal number of epochs to be around 10 by experimentation. I used an adam optimizer so that manually training the learning rate wasn't necessary.
