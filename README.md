# First-steps-with-CNN: Classification-on-the-CIFAR-10-Dataset

## INTRODUCTION

Hello world, today I'm coming with my first repository on this plateform. The goal is to try a simple classification task with the CIFAR-10 Dataset in order to understand better how CNNs work. The CIFAR 10 Dataset is a set of 60 000 tiny images (32 x 32) coming from 10 various classes : airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truck as we can see down there. So together, we'll try to develop a deep learning model able to classify a new image in the right class.

<p float="left">
    <img src="Images/sample.PNG" width="425"/> 
</p>

## PROJECT
This repository contains 2 main files: 

¤ CIFAR-10 Classification which is a notebook that describes the whole process: how to collect data, how to preprocess those data and create a dataset, how to build, train and test our deep learning model.

¤ visual callebacks which is a python file that contains the callbacks I used to display the loss and accuracy curves as well as the confusion matrix at the end of each epoch. This is a great tool to see how our CNN behaves with time.

For more information, check the article I wrote on medium by clicking [here]

## RESULTS
Our model has an accuracy of 0.76 on the training set and 0.72 on the test set. This is not the best result in the world but it's a good start. Here are some results for some test images.

