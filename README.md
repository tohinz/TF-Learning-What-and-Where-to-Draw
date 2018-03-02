# TF-Learning-What-and-Where-to-Draw
Tensorflow implementation of the NIPS 2016 paper ["Learning What and Where to Draw"](http://papers.nips.cc/paper/6111-learning-what-and-where-to-draw) by Scott Reed et al.

We use a modified MNIST data set for the experiment. The images are of size 64 x 64 pixels, where the mnist images situated randomly within the image. For the generation of the data set we use the (slightly modified) code from [Adnan Akhundov](https://github.com/aakhundov/tf-attend-infer-repeat).

## Acknowledgements
* The code for the spatial transformer network is taken from [Kevin Zakka](https://github.com/kevinzakka/spatial-transformer-network)
* The code for generating the Multi-MNIST data set images is taken from [Adnan Akhundov](https://github.com/aakhundov/tf-attend-infer-repeat)

## Requirements
* Tensorflow 1.5.0

## To Create the Data Set
* run `python multi_mnist.py`
* default values are set to 120 000 images, resolution of (64, 64, 1), with one (possibly stretched) MNIST digit placed somewhere in the image
* data gets stored in folder "positional_mnist_data/1.tfrecords"

## To Run the Experiment
* run `python lwawtd.py`
* to visualize the progress: `tensorboard --logdir log_cir/lwawtd/`
