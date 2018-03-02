# Tensorflow Version: 1.5.0

import os
import sys
import numpy as np
import datetime
import dateutil.tz
import argparse
from shutil import copyfile
import math

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow.python.layers import utils

from spatial_transformer import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="Direcotry to the model weights", type=str)
parser.add_argument("--iteration", help="Which model weights should be used", type=int, default=20000)
parser.add_argument("--digit", help="Images of which digit should be generated", type=int)
parser.add_argument("--draw_bbox", help="Draw the bounding box into the generated images", action="store_true")
args = parser.parse_args()


def read_hyperparameters(path):
    reader = csv.reader(open(path+"/hyperparameters"+args.model_dir[-20:]+".csv", "rb"))
    dict = {}
    for row in reader:
        k, v = row
        dict[k] = v

    return dict

hp_dict = read_hyperparameters(args.model_dir)


# hyperparameters
Z_DIM = hp_dict["num_z"]                 # dimensionality of the z vector (input to G, incompressible noise
LABEL_DIM = 10                           # dimensionality of the label vector (axis 1)
MODEL = args.model + "iteration.ckpt-" + str(args.iteration)
if args.digit is not None:
    DIGIT = args.digit


now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

log_dir = "generated_images/" + str(sys.argv[0][:-3]) + "/lwawtd_" + timestamp

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# save executed file
copyfile(sys.argv[0], log_dir + "/" + sys.argv[0])

with open(log_dir + "/hyperparameters_"+timestamp+".csv", "wb") as f:
    for arg in args.__dict__:
        f.write(arg + "," + str(args.__dict__[arg]) + "\n")

weight_init = None
# activation functions for the Generator and Discriminator
activations = {"elu" : tf.nn.elu, "relu": tf.nn.relu, "lrelu": tf.nn.leaky_relu}
g_activation = activations[hp_dict["g_activation"]]
d_activation = activations[hp_dict["d_activation"]]


# placeholder variables
phase = tf.placeholder(tf.bool, name='phase') # training or inference
Y_ = tf.placeholder(tf.float32, shape=[None, LABEL_DIM], name="Y") # labels for generated images
z = tf.placeholder(tf.float32, shape=[None, Z_DIM], name="z") # noise, input to G
bbox_ = tf.placeholder(tf.float32, shape=[None, 4], name="bbox_") # bounding boxes of generated images


def conv2d_bn_act(inputs, filters, kernel_size, kernel_init, activation, strides, padding="SAME"):
    """
    Shortcut for a module of convolutional layer, batch normalization and possibly adding of Gaussian noise.
    :param inputs: input data
    :param filters: number of convolutional filters
    :param kernel_size: size of filters
    :param kernel_init: weight initialization
    :param activation: activation function (applied after batch normalization)
    :param strides: strides of the convolutional filters
    :param padding: padding in the conv layer
    :return: output data after applying the conv layer, batch norm, activation function and possibly Gaussian noise
    """
    _tmp = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            kernel_initializer=kernel_init, activation=None, strides=strides, padding=padding)
    _tmp = tf.contrib.layers.batch_norm(_tmp, center=True, scale=True, is_training=phase, fused=True)
    _tmp = activation(_tmp)

    return _tmp


def deconv2d_bn_act(inputs, filters, kernel_size, kernel_init, activation, strides, padding="SAME"):
    """
        Shortcut for a module of transposed convolutional layer, batch normalization.
        :param inputs: input data
        :param filters: number of convolutional filters
        :param kernel_size: size of filters
        :param kernel_init: weight initialization
        :param activation: activation function (applied after batch normalization)
        :param strides: strides of the convolutional filters
        :param padding: padding in the conv layer
        :return: output data after applying the transposed conv layer, batch norm, and activation function
        """
    _tmp = tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=kernel_size,
                            kernel_initializer=kernel_init, activation=None, strides=strides, padding=padding)
    _tmp = tf.contrib.layers.batch_norm(_tmp, center=True, scale=True, is_training=phase, fused=True)
    _tmp = activation(_tmp)

    return _tmp


def dense_bn_act(inputs, units, activation, kernel_init):
    """
        Shortcut for a module of dense layer, batch normalization and possibly adding of Gaussian noise.
        :param inputs: input data
        :param units: number of units
        :param activation: activation function (applied after batch normalization)
        :param kernel_init: weight initialization
        :return: output data after applying the dense layer, batch norm, activation function and possibly Gaussian noise
        """
    _tmp = tf.layers.dense(inputs=inputs, units=units, activation=None, kernel_initializer=kernel_init)
    _tmp = tf.contrib.layers.batch_norm(_tmp, center=True, scale=True, is_training=phase, fused=True)
    _tmp = activation(_tmp)

    return _tmp


# Discriminator
def discriminate(image_input, label, bounding_box):
    with tf.variable_scope("d_net", reuse=tf.AUTO_REUSE):
        # image discriminator, initial processing
        input = conv_cond_concat(image_input, label)
        d_x_conv_0 = conv2d_bn_act(inputs=input, filters=32, kernel_size=4, kernel_init=weight_init,
                                   activation=d_activation, strides=2)
        d_x_conv_1 = conv2d_bn_act(inputs=d_x_conv_0, filters=64, kernel_size=4, kernel_init=weight_init,
                                   activation=d_activation, strides=2)

        ####################################################
        # global pathway
        d_x_conv_global_0 = conv2d_bn_act(inputs=d_x_conv_1, filters=64, kernel_size=4, kernel_init=weight_init,
                                   activation=d_activation, strides=2)

        d_x_conv_global_1 = conv2d_bn_act(inputs=d_x_conv_global_0, filters=128, kernel_size=4, kernel_init=weight_init,
                                   activation=d_activation, strides=2)
        shp = [int(s) for s in d_x_conv_global_1.shape[1:]]
        d_x_conv_global_1 = tf.reshape(d_x_conv_global_1, [-1, shp[0] * shp[1] * shp[2]])

        ####################################################
        # local pathway
        # reshape bounding box to (16, 16) resolution
        transf_matri = tf.map_fn(tf_compute_transformation_matrix, bounding_box)
        local_input = spatial_transformer_network(d_x_conv_1, transf_matri, (16, 16))
        d_x_conv_local_0 = conv2d_bn_act(inputs=local_input, filters=64, kernel_size=4, kernel_init=weight_init,
                                   activation=d_activation, strides=2)
        d_x_conv_local_1 = conv2d_bn_act(inputs=d_x_conv_local_0, filters=128, kernel_size=4, kernel_init=weight_init,
                                         activation=d_activation, strides=2)
        shp = [int(s) for s in d_x_conv_local_1.shape[1:]]
        d_x_conv_local_1 = tf.reshape(d_x_conv_local_1, [-1, shp[0] * shp[1] * shp[2]])

        ####################################################
        # final discriminator
        final_input = tf.concat((d_x_conv_global_1, d_x_conv_local_1), axis=1)
        d_final_dense = dense_bn_act(inputs=final_input, units=512, activation=d_activation, kernel_init=weight_init)
        d_final_pred = tf.layers.dense(inputs=d_final_dense, units=1,
                                       activation=tf.nn.sigmoid, kernel_initializer=weight_init)

        return d_final_pred

# Generator
def generate(noise_input, label, bounding_box):
    input = tf.concat((noise_input, label), axis=1)
    with tf.variable_scope("g_net", reuse=tf.AUTO_REUSE):
        # reshape input
        g_dense_0 = dense_bn_act(inputs=input, units=2048, activation=g_activation, kernel_init=weight_init)
        g_dense_0 = tf.reshape(g_dense_0, [-1, 4, 4, 128])

        ####################################################
        # global pathway
        g_conv_global_0 = deconv2d_bn_act(inputs=g_dense_0, filters=64, kernel_size=4, kernel_init=weight_init,
                      activation=g_activation, strides=2)
        g_conv_global_1 = deconv2d_bn_act(inputs=g_conv_global_0, filters=64, kernel_size=4, kernel_init=weight_init,
                                 activation=g_activation, strides=2)

        ####################################################
        # local pathway
        g_conv_local_0 = deconv2d_bn_act(inputs=g_dense_0, filters=64, kernel_size=4, kernel_init=weight_init,
                      activation=g_activation, strides=2)
        g_conv_local_1 = deconv2d_bn_act(inputs=g_conv_local_0, filters=64, kernel_size=4, kernel_init=weight_init,
                                 activation=g_activation, strides=2)
        # reshape to bounding box
        transf_matri = tf.map_fn(tf_compute_transformation_matrix_inverse, bounding_box)
        g_conv_local_1 = spatial_transformer_network(g_conv_local_1, transf_matri, (16, 16))

        ####################################################
        # final pathway
        final_input = tf.concat((g_conv_global_1, g_conv_local_1), axis=3)
        g_conv_final = deconv2d_bn_act(inputs=final_input, filters=32, kernel_size=4, kernel_init=weight_init,
                                   activation=g_activation, strides=2)
        g_conv_final_2 = deconv2d_bn_act(inputs=g_conv_final, filters=32, kernel_size=4, kernel_init=weight_init,
                                       activation=g_activation, strides=2)
        g_conv_out = tf.layers.conv2d_transpose(inputs=g_conv_final_2, filters=1, kernel_size=4,
                                                activation=tf.nn.sigmoid, padding='SAME',
                                                strides=1, kernel_initializer=weight_init)

        return g_conv_out


def sample_bbox(mb_size):
    pos_box_x = np.zeros((mb_size, 1))
    for idx in range(100):
        pos_box_x[idx, 0] = (idx % 10 + 1) * 4

    pos_box_y = np.zeros((mb_size, 1))
    for idx in range(10):
        pos_box_y[idx * 10:idx * 10 + 10, 0] = (idx + 1) * 4

    scale_box_x = np.zeros((mb_size, 1))
    for idx in range(100):
        scale_box_x[idx, 0] = (idx % 10 + 1) + 10

    scale_box_y = np.zeros((mb_size, 1))
    for idx in range(10):
        scale_box_y[idx * 10:idx * 10 + 10, 0] = (idx + 1) + 10

    boxes = np.concatenate((pos_box_x, pos_box_y, scale_box_x, scale_box_y), axis=1)
    return boxes.astype(np.float32)


def create_image(images, title, bbox):
    fig = plt.figure(figsize=(0.68 * 10, 0.68 * 10))
    gs1 = gridspec.GridSpec(10, 10, wspace=0.0, hspace=0.0)
    for idx1 in range(10):
        for idx2 in range(10):
            img = np.reshape(images[idx1*10+idx2], [64, 64])
            if args.draw_bbox:
                x = int(bbox[idx1*10+idx2, 0])
                y = int(bbox[idx1*10+idx2, 1])
                w = int(bbox[idx1*10+idx2, 2])
                h = int(bbox[idx1*10+idx2, 3])
                img[y, x:x + w] = 1
                img[y + h, x:x + w] = 1
                img[y:y + h, x] = 1
                img[y:y + h, x + w] = 1
            ax = plt.subplot(gs1[idx1 * 10 + idx2])
            img = np.pad(img, [[2,2], [2,2]], mode="constant", constant_values=1)
            ax.imshow(img, cmap='Greys_r')
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout(pad=0)
    fig.savefig(log_dir+"/"+title+".png")


f, axarr = plt.subplots(2, 10)
plt.tick_params(
    axis='x',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom='off',  # ticks along the bottom edge are off
    top='off',  # ticks along the top edge are off
    labelbottom='off')
plt.tick_params(
    axis='y',  # changes apply to the x-axis
    which='both',  # both major and minor ticks are affected
    bottom='off',  # ticks along the bottom edge are off
    top='off',  # ticks along the top edge are off
    labelbottom='off')

# image generated by generator G
X_hat = generate(noise_input=z, label=Y_, bounding_box=bbox_)

print("Loading model")
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, MODEL)

if args.digit is not None:
    box_generated = sample_bbox(100)
    label = np.zeros((100, 10)).astype(np.float32)
    label[:, args.digit] = 1.

    _z = sample_z(100, Z_DIM)

    imgs = sess.run(X_hat, feed_dict={phase: 0, Y_: label, z: _z, bbox_: box_generated})

    create_image(imgs, "digit_" + str(idx), box_generated)
else:
    box_generated = sample_bbox(100)
    for idx in range(10):
        label = np.zeros((100, 10)).astype(np.float32)
        label[:, idx] = 1.

        _z = sample_z(100, Z_DIM)

        imgs = sess.run(X_hat, feed_dict={phase: 0, Y_: label, z: _z, bbox_: box_generated})

        create_image(imgs, "digit_"+str(idx), box_generated)
