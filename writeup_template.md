#**Behavioral Cloning** 

##Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the nvidia model (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I tested with other models like comma.ai but found out the nvidia got me better results. 
I allowed to create new models and to choose the one to use a parameter is passed to the train function.

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. 20% of the data was using on the validation set.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used data provided by Udacity plus data I generated myself which consisted on two laps in the normal direction, one in the opposite direction and some recovery from the sides but I ended up using only data provided.

I used a combination of center lane driving, recovering from the left and right sides of the road. At some stage I tried randomly selecting with image to choose (center, left or right) but ended up using all of them as I got better results. The correction to apply is passed as a parameter to the train function and found that 0.08 was the value with better results.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to collect some data by driving the simulator myself and add that data to the one provided by Udacity.

I used a convolution neural network model based on the nvidia model. I thought this model might be appropriate because it was created for the same purpose and found out it works pretty well.

To avoid preprocessing data in memory all at once I used a generator with a batch size of 32. In the generator I augmented the dataset by flipping each image horizontally and inverting the steering angle. I also made use of all three images (center, left and right) applying a correction for the side images. I tested at some stage to randomize what image to use so the examples are more normally distributed but I ended up using all images as I got better results.

Images size was 160x320 and I used a keras cropping layer to remove portions of the image that are not the road. I removed 70 pixels from the top of the image and 25 from the bottom ending up with an image size of 65x320. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 20% of the data was used for the validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes 

```sh
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_15 (Lambda)               (None, 160, 320, 3)   0           lambda_input_15[0][0]            
____________________________________________________________________________________________________
cropping2d_15 (Cropping2D)       (None, 65, 320, 3)    0           lambda_15[0][0]                  
____________________________________________________________________________________________________
convolution2d_63 (Convolution2D) (None, 31, 158, 24)   1824        cropping2d_15[0][0]              
____________________________________________________________________________________________________
convolution2d_64 (Convolution2D) (None, 14, 77, 36)    21636       convolution2d_63[0][0]           
____________________________________________________________________________________________________
convolution2d_65 (Convolution2D) (None, 5, 37, 48)     43248       convolution2d_64[0][0]           
____________________________________________________________________________________________________
convolution2d_66 (Convolution2D) (None, 3, 35, 64)     27712       convolution2d_65[0][0]           
____________________________________________________________________________________________________
convolution2d_67 (Convolution2D) (None, 1, 33, 64)     36928       convolution2d_66[0][0]           
____________________________________________________________________________________________________
flatten_15 (Flatten)             (None, 2112)          0           convolution2d_67[0][0]           
____________________________________________________________________________________________________
dense_49 (Dense)                 (None, 100)           211300      flatten_15[0][0]                 
____________________________________________________________________________________________________
dense_50 (Dense)                 (None, 50)            5050        dense_49[0][0]                   
____________________________________________________________________________________________________
dense_51 (Dense)                 (None, 10)            510         dense_50[0][0]                   
____________________________________________________________________________________________________
dense_52 (Dense)                 (None, 1)             11          dense_51[0][0]                   
====================================================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
____________________________________________________________________________________________________
```


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
