# **Traffic Sign Recognition** 

## Writeup

Please note that the final submission file is **Traffic_Sign_Classifier.ipynb** in the main directory.
There is also a html file by the same name included in this submission. The new test images downloaded from the web are in the `./test_images` directory and all the images in the writeup 
can be found in the directory `./output_images`

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/all_classes.png "Visualization of all classes in the dataset"
[image2]: ./output_images/data_distro.png "Data distribution"
[image3]: ./output_images/grayscale.png "Grayscaling"
[image4]: ./output_images/CLAHE.png "CLAHE"
[image5]: ./output_images/data_aug.png "Data Augmentation"
[image6]: ./test_images/1.jpg "Traffic Sign 1"
[image7]: ./test_images/2.jpg "Traffic Sign 2"
[image8]: ./test_images/3.jpg "Traffic Sign 3"
[image9]: ./test_images/4.jpg "Traffic Sign 4"
[image10]: ./test_images/5.jpg "Traffic Sign 5"
[image11]: ./output_images/top_5_0.png "Top 5"
[image12]: ./output_images/top_5_1.png "Top 5"
[image13]: ./output_images/top_5_2.png "Top 5"
[image14]: ./output_images/top_5_3.png "Top 5"
[image15]: ./output_images/top_5_4.png "Top 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is **34799**.
* The size of the validation set is **4410**.
* The size of test set is **12630**.
* The shape of a traffic sign image is **(32, 32, 3)**.
* The number of unique classes/labels in the data set is **43**.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I first looked at a sample from each class in the dataset as shown in 
`./ouput_images/all_classes.png`

![Display all classes][image1]

To visualize the number of samples per classes, I created a bar chart.

![Bar chart of class distribution][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduces the number of parameters
as the input is of a lower dimension. I wanted to try out with grayscale image first for simplicty.

Here is an example of a traffic sign image before and after grayscaling.

![Grayscaling][image3]

From my initial exploration of the dataset, I realized that the images had varied lighting and contrast,
which can lead to poor performance. To improve the contrast in the image, I opted to use Contrast Limited
Adaptive Histogram Equalization (CLAHE). I opted for CLAHE as opposed to normal histogram equalization, as
vanilla histogram equalization considers the global contrast of the image and can often lead to loss of information
due to over-brightness. I saw a spike in performance on both the training and validation set after including CLAHE.
Here is an example of a traffic sign image after applying CLAHE.

![CLAHE][image4]

As a last step, I normalized the image data because so that the variance in the image data
is not too skewed.

I decided to generate additional data because as there is high variance in the number of samples
per class in the training dataset.

I experimented with random flips both left-right, and up-down), rotation and translation. These are illustrated
below for a sample image: 

![Data Augmentation][image5]

Please note that I simply experimented with these techniques and did not actually perform data augmentation. The
reason being not all augmentation techniques can be applied to all signs. For instance, flip or rotate operation 
when applied to a **turn left** sign, can cause it to look identical to the **turn right** sign. Thus,
such data augmentation techniques must be specific to each class. Since, I already achieved reasonable
performance from preprocessing images, I did not perform any data augmentation on the training set. A potential
improvement / future work would involve performing data augmentation specifically to those classes that have
few samples in the dataset. Doing so, can improve the performance of the classifier. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers (**b** indicates batch size):

| Layer         		|     Description	        					| Input       |Output      |
|:---------------------:|:---------------------------------------------:| -----------:|-----------:|
| Input         		| Batch (b) of 32x32x1 preprocessed images  	|		      |            |
| Convolution 5x5     	| 6 filters,1x1 stride, VALID padding       	|(bx32x32x1)  |(bx28x28x6) |
| RELU					| ReLU nonlinear activation						|(bx28x28x6)  |(bx28x28x6) |
| Max pooling	      	| 2x2 stride                     				|(bx28x28x6)  |(bx14x14x6) |
| Convolution 5x5     	| 16 filters,1x1 stride, VALID padding       	|(bx14x14x6)  |(bx10x10x16)|
| RELU					| ReLU nonlinear activation						|(bx10x10x16) |(bx10x10x16)|
| Max pooling	      	| 2x2 stride                     				|(bx10x10x16) |(bx5x5x16)  |
| Flatten       	    | Flatten conv outputs 							|(bx5x5x16)   |(bx400)     |
| Fully connected		| FC layer of 120 units with dropout 			|(bx400)      |(bx120)     |
| RELU					| ReLU nonlinear activation						|(bx120)      |(bx120)     |
| Fully connected		| FC layer of 84 units with dropout 			|(bx120)      |(bx84)      |
| RELU					| ReLU nonlinear activation						|(bx84)       |(bx84)      |
| Fully connected		| FC layer of 43 (n_classes) units          	|(bx84)       |(bx43)      |

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the same LeNet architecture from the Convolutional Networks module. I used the Adam Optimizer.
I experimented with different learning rates (0.01, 0.001, 0.005), batch sizes and number of epochs. After some trial and error,
I chose the following hyperparameters:
- Learning rate: 0.005
- Number of epochs: 20
- Batch Size: 100

I obtained the expected performance metric (0.93 accuracy) on validation set by tuning these
hyperparameters.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of **0.981**
* validation set accuracy of **0.953** 
* test set accuracy of **0.923**

I chose to use the same LeNet architecture for the Traffic Sign Classification Task. I chose this
architecture as this architecture has already been utilized in Dr.LeCun's paper and they show
very strong results. 

Convolutional networks perform very well on images, as the convolution operation leverages the structure of the image,
and is invariant to translations in the image. Moreover by using convolutional networks, we reduce the
number of parameters required to learn the task as compared to a Multi layer perceptron.

I tuned the hyperparameters - learning rate, batch size
and number of epochs. I tried varying the hyperparameters one at a time in small increments / decrements
until desired performance was reached.

Further, I added dropout layers for the fully connected networks as a means to 
avoid overfitting on the training dataset. I noticed that after adding the dropout layers, the
difference in accuracy between the training set and the validation set was reduced, indicating better
overall performance.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Image 1][image6] ![Image 2][image7] ![Image 3][image8] 
![Image 4][image9] ![Image 5][image10]

- The first image might be difficult to classify because there is a black background due to
improper cropping along with the blue sky. 
- The second image might be difficult to classify because the sign is tilted away from view.
- The third image might be difficult to classify because the there is additional font on the
image as it is downloaded from the web.
- The fourth image might be difficult to classify because of th background with the clouds. It
is more straightforward than the other images in the test samples.
- The fifth image might difficult to classify because of the change in color.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road Work      		| Road Work   									| 
| Stop       			| General Caution 								|
| 60 km/h				| 60 km/h										|
| Right of way     		| Right of way					 				|
| Keep right			| Keep right      							    |


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of given images,
which is 92.7%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 23rd cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a road work sign (probability of 0.6), and the image does contain a stop sign.
The model has high difficulty in predicting the stop sign. This might be because of the orientation of the sign.
For the last three images, the model gets the prediction right by a substantial margin - 60km/h speed limit, 
right of way at next intersection, and keep right. The top 5 probabilities along with the image are shown below:

![Top 5][image11]
![Top 5][image12]
![Top 5][image13]
![Top 5][image14]
![Top 5][image15]


