# IEEE SPS ML Tutorial

This tutorial covers on how to train a CNN classifier using MNIST dataset. 
MNIST data contains around 70000 images, out of which 60000 are used for training and 10000 are used for testing/evaluation.
Each image is a hand written digit (0-9), 10 classes of size 28x28 (Grayscale Image).

## Pre-requisites:

* Pytorch installed
* Knowledge on python/numpy is a plus.

## Guide to try out the tutorial:

Most of the code has been already written for you. 
This tutorial guides you on how to use pytorch APIs to easily train a CNN classifier using MNIST Dataset. 
Once you clone the repository, the first step is to update the code to pre-process the MNIST dataset and convert into a meaningful format that pytorch understands. 

The files contains comments which are an useful guide to properly train a model.

* The Data pre-processing code is present in mnist_data.py 
* Creation of the model is present in model.py
* The actual training is done in main.py. Update the hyperparameters, and the training code to complete the steps.

Once you finish the implementation use the following commands to test and check if you are getting proper accuracy.  

* To run on a CPU:
```
python main.py --batch-size 8 --epochs 2 --use-cpu --save-model 
```

* To run on a GPU:
```
python main.py --batch-size 8 --epochs 2 --save-model
```

## Solutions
Please navigate to the branch ```solutions``` in this repository to avail the code for the CNN classifier.

