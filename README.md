# Face Recognition with OpenCV and Python
The project has two seperate folders that are independent of one another.

## Face Detection

This folder demonstrates the models built using haarcascade_frontalface_default.xml, a built-in cascade classifier in openCV that detect faces in an image and a video respectively and highlight the faces detected within a rectangle.

## Face Recognition

This folder has 3 parts
1) The Training and Validation data used to train the model and then test the model.
2) Data Labeling using haar cascade and os. Building model using LBPHFaceRecognizer_create() of opencv-contrib-python and training on labelled data.
3) Testing model using validation data.

### Required Modules

* opencv-contrib-python: This package contains extra modules added on top of opencv package which enhance the functionality and speed of opencv package.
* os: We will use this Python module to read our training directories and file names.
* numpy: We will use this module to convert Python lists to numpy arrays as OpenCV face recognizers accept numpy arrays.

**Note:** I am using not using opencv-python but opencv-contrib-python package available on pypi.org that can be installed using pip. 
