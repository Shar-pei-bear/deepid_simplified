Project name: deepid_simplified

Description: a simplified implementation of deepid described in the paper Hybrid Deep Learning for Face Verification. paper link:http://www.ee.cuhk.edu.hk/~xgwang/papers/sunWTpami16.pdf

Table of Contents:
  image_crop.py：crop five face regions

  split_CNN_RBM.py：split the cropped images between CNN and RBM training.

  split_valid_test.py： split the rest of the images between valid set and test set.

  vec_tfrecord_CNN.py: transform the image data for CNN into tfrecords.

  vec_tfrecord_RBM.py: transform the image data for RBM into tfrecords.

  CNN.py：Train and evaluate the accuracy of the CNN.

  CNN_RBM.py：Train and evaluate the accuracy of the RBM network and the overall network.

  My_function.py: user defined functions

  interface.py：implementing an interface function named FaceVerification

  sample.py: an example using FaceVerification

Installation: several python wheels are needed: tensorflow, numpy, tensorflow, numpy,dlib, pillow.

Usage: see an example of usage in sample.py

License: MIT license
