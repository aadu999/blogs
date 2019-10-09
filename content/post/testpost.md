---
title: "Face Recognition using KNN"
date: 2019-10-05
categories:
- Machine Learning
- Classification
- KNN
tags:
- facial recognition
- knn
- opencv
- machine learning
- application of machine learning
keywords:
- disqus
- google
- gravatar
autoThumbnailImage: true
thumbnailImagePosition: "top"
thumbnailImage: https://internetofbusiness.com/wp-content/uploads/2018/06/facial-recognition.jpg
coverImage: /images/onecover.jpeg
metaAlignment: center
---
This is a small attempt to make a facial recognition system using K Nearest Neighbor algorithm. We will be using OpenCV, Numpy and some other python packages. <!--more--> Initially we've to create a KNN Classifier from the scratch, Those who don't Know much about KNN classifier please read [this](https://google.com).

### Package Installation
kindly make sure you've installed Numpy and Open CV else please use the snippet below.

```shell
pip3 install numpy opencv-python
```
### Project Breakdown
1. Using webcam record the images
2. Detect faces in the image
3. Pass the facial data to a KNN model
4. Get live prediction of the faces from the model

If you are new to OpenCV please read the documentation [here](https://docs.opencv.org/3.4.7/). In order to capture the image we need to create a python script; lets name it as capture.

```python
import cv2

cap = cv2.VideoCapture(0)

while True:
	ret,image = cap.read()
	if ret == True:
		cv2.imshow('frame',image)
		if cv2.waitKey(1) == 27:
			cv2.imwrite('savedImage.jpg',image)
			cv2.destroyAllWindows()
			break
```
After capturing the image with a face in it, we need to extract face out of the image. Face detection uses _**classifiers**_, which are algorithms that detects what is either **face(1)** or **not a face(2)** in an image. OpenCV uses two types of classifiers; LBP (Local Binary Points) and Haar Cascades. We will be using Haar Cascades. 

## What are Haar Cascades ?
A Haar Cascade is based on “Haar Wavelets” which Wikipedia defines as:

> **A sequence of rescaled “square-shaped” functions which together form a wavelet family or basis.**

It is based on the Haar Wavelet technique to analyze pixels in the image into squares by function. This uses machine learning techniques to get a high degree of accuracy from what is called “training data”. This uses “integral image” concepts to compute the “features” detected. Haar Cascades use the Adaboost learning algorithm which selects a small number of important features from a large set to give an efficient result of classifiers.

{{< figure src="/images/knn-post.png" alt="Haar cascade Descriptive Illustration" >}}

Pre-trained Haar models for frontal face is available to [download here](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml), hence we are going to use that.

```python

import cv2

imagePath = "/Users/adarsh/Desktop/Works/threemen.jpg"

cascadePath = "/Users/adarsh/Desktop/Works/haarcascade_frontalface_default.xml"

#create the face cascade classifier using the xml file
faceCascade = cv2.CascadeClassifier(cascadePath)

#read the image
image = cv2.imread(imagePath)

#convert the image in to gray color
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray= cv2.equalizeHist(gray)

# The face or faces in an image are detected
# This section requires the most adjustments to get accuracy on face being detected.
#the first parameter is the image,then the scale factor and the minimum number of neighbors

faces = faceCascade.detectMultiScale(gray,1.2,5)

for (x,y,w,h) in faces:
    #Here we are drawing rectangle over the faces detected using x,y co-ordinates and w,h height and width in blue color
    image = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow('img',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
{{< figure src="/images/haar-output.jpg" alt="FaceDetection using Haar cascade frontal face default" >}}

Now that we've detected the face, we need to store it somewhere. So here we are going to acquire the images using a webcam and we will store the captured faces in a numpy npz file, it save several arrays into a single file in uncompressed .npz format. We are going to create a face data generator file as mentioned below.



