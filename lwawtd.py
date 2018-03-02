# Tensorflow Version: 1.5.0

import os
import sys
import numpy as np
import datetime
import dateutil.tz
import argparse
from shutil import copyfile
import math

import tensorflow as tf
from tensorflow.python.layers import utils
import tensorflow.contrib.eager as tfe
from tensorflow.python.client import timeline

from spatial_transformer import *

from utils import *
from tb_visualization import *

parser = argparse.ArgumentParser()
parser.add_argument("--max_iter", help="Maximum number of iterations", type=int, default=20000)
parser.add_argument("--batch_size", help="The size of the minibatch", type=int, default=64)
parser.add_argument("--lr_d", help="Discriminator Learning Rate", type=float, default=1e-4)
parser.add_argument("--lr_g", help="Generator/Encoder Learning Rate", type=float, default=1e-4)
parser.add_argument("--beta1_g", help="Generator Beta 1 (for Adam Optimizer)", type=float, default=0.5)
parser.add_argument("--beta1_d", help="Discriminator Beta 1 (for Adam Optimizer)", type=float, default=0.5)
parser.add_argument("--num_z", help="Number of noise variables", type=int, default=100)
parser.add_argument("--d_activation", help="Activation function of Discriminator", type=str, default="lrelu")
parser.add_argument("--g_activation", help="Activation function of Generator", type=str, default="relu")
parser.add_argument("--dataset", help="Path to the data set you want to use", type=str,
                    default="positional_mnist_data/1.tfrecords")
parser.add_argument("--weight_init", help="Values for weight init method", type=int, default=0)
parser.add_argument("--boundaries", help="Boundaries for LR decrease", type=int, default=0)
parser.add_argument("--values", help="Values for LR decrease", type=int, default=0)
parser.add_argument("--g_train", help="How often generator is trained during each iteration", type=int, default=1)
parser.add_argument("--eager", help="If eager execution is enabled", action="store_true")
parser.add_argument("--timeline", help="If a timeline is created", action="store_true")
parser.add_argument("--metadata", help="If metadata is logged (requires libcupti)", action="store_true")
args = parser.parse_args()

if args.eager:
    tfe.enable_eager_execution()


# hyperparameters
BATCH_SIZE = args.batch_size        # training batch size
MAX_ITER = args.max_iter            # maximum number of iterations
IMG_WIDTH, IMG_HEIGHT = 64, 64      # image dimensions
IMG_CHANNELS = 1                    # image channels
LR_D = args.lr_d                    # learning rate discriminator
LR_G = args.lr_g                    # learning rate generator
BETA1_D = args.beta1_d              # beta1 value for Adam optimizer (discriminator)
BETA1_G = args.beta1_g              # beta1 value for Adam optimizer (generator)

Z_DIM = args.num_z                  # dimensionality of the z vector (input to G, incompressible noise)
LABEL_DIM = 10                      # dimensionality of the label vector (axis 1)


now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

log_dir = "log_dir/" + str(sys.argv[0][:-3]) + "/lwawtd_" + timestamp

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# save executed file
copyfile(sys.argv[0], log_dir + "/" + sys.argv[0])

with open(log_dir + "/hyperparameters_"+timestamp+".csv", "wb") as f:
    for arg in args.__dict__:
        f.write(arg + "," + str(args.__dict__[arg]) + "\n")


# activation functions for the Generator and Discriminator
activations = {"elu" : tf.nn.elu, "relu": tf.nn.relu, "lrelu": tf.nn.leaky_relu}
g_activation = activations[args.g_activation]
d_activation = activations[args.d_activation]


if 0 <= args.weight_init <= 3:
    if args.weight_init == 0:
        factor = 2.0
        mode = 'FAN_IN'
        uniform = False
    elif args.weight_init == 1:
        factor = 1.0
        mode = 'FAN_IN'
        uniform = True
    elif args.weight_init == 2:
        factor = 1.0
        mode = 'FAN_AVG'
        uniform = True
    elif args.weight_init == 3:
        factor = 1.0
        mode = 'FAN_AVG'
        uniform = False
    weight_init = tf.contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
elif args.weight_init == 4:
    weight_init = tf.truncated_normal_initializer(stddev=0.02)


def read_and_decode(fqueue, batch_size, num_threads):
    reader = tf.TFRecordReader()
    key, value = reader.read(fqueue)

    features = tf.parse_single_example(
        value,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'labels': tf.FixedLenFeature([], tf.string),
            'positions': tf.FixedLenFeature([], tf.string),
            'boxes': tf.FixedLenFeature([], tf.string)
        }
    )

    batch = tf.train.shuffle_batch(
        [
            tf.reshape(tf.decode_raw(features['image'], tf.float32), [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS]),
            tf.reshape(tf.one_hot(tf.decode_raw(features['labels'], tf.int32), LABEL_DIM), [LABEL_DIM]),
            tf.reshape(tf.concat((tf.reshape(tf.decode_raw(features['positions'], tf.int32), [1, 2]),
                       tf.reshape(tf.decode_raw(features['boxes'], tf.int32), [1, 2])), axis=1), [4])
        ],
        batch_size=batch_size,
        capacity=10000+num_threads*batch_size*10,
        min_after_dequeue=10000,
        num_threads=num_threads,
    )

    return batch


def sample_real_data(batch_size, queue=args.dataset):
    filename_queue = tf.train.string_input_producer([queue], num_epochs=None)
    train_data, labels, bbox = read_and_decode(filename_queue, batch_size, 4)

    return train_data, labels, tf.cast(bbox, tf.float32)


# placeholder variables
phase = tf.placeholder(tf.bool, name='phase') # training or inference

Y_ = tf.placeholder(tf.float32, shape=[None, LABEL_DIM], name="Y") # labels
z = tf.placeholder(tf.float32, shape=[None, Z_DIM], name="z") # incompressible noise, input to G
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
        g_conv_out = tf.layers.conv2d_transpose(inputs=g_conv_final_2, filters=IMG_CHANNELS, kernel_size=4,
                                                activation=tf.nn.sigmoid, padding='SAME',
                                                strides=1, kernel_initializer=weight_init)

        return g_conv_out


# input pipeline
with tf.device('/cpu:0'):
    # sample training data
    X, Y, bbox = sample_real_data(BATCH_SIZE)

# image generated by generator G
X_hat = generate(noise_input=z, label=Y_, bounding_box=bbox_)

# prediction of D for real images with bounding box
D_enc = discriminate(image_input=X, label=Y, bounding_box=bbox)
# prediction of D for generated images with randomly sampled noise
D_gen = discriminate(image_input=X_hat, label=Y_, bounding_box=bbox_)

# Discriminator loss
D_loss = -tf.reduce_mean(log(D_enc) + log(1 - D_gen))
# Generator loss
G_loss = -tf.reduce_mean(log(D_gen))

# Define the optimizers for D, G, and E
all_vars = tf.trainable_variables()
theta_G = [var for var in all_vars if var.name.startswith('g_')]
theta_D = [var for var in all_vars if var.name.startswith('d_')]


update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
boundaries = {0: [20000, 20000, 20000], 1: [5000, 10000, 15000], 2: [2500, 5000, 10000], 3: [10000, 15000, 20000]}
boundaries = boundaries[args.boundaries]

values_D = {0: [LR_D, LR_D / 2.0, LR_D / 4.0, LR_D / 8.0], 1: [LR_D, LR_D / 5.0, LR_D / 10.0, LR_D / 20.0]}
values_D = values_D[args.values]

values_G = {0: [LR_G, LR_G / 2.0, LR_G / 4.0, LR_G / 8.0], 1: [LR_G, LR_G / 5.0, LR_G / 10.0, LR_G / 20.0]}
values_G = values_G[args.values]
with tf.control_dependencies(update_ops):
    # Ensures that we execute the update_ops before performing the train_step, important for updating
    # the batch normalization parameters
    global_step_d = tf.Variable(0, trainable=False)
    global_step_g = tf.Variable(0, trainable=False)

    lr_dis = tf.train.piecewise_constant(global_step_d, boundaries, values_D)
    D_solver = (tf.train.AdamOptimizer(learning_rate=lr_dis, beta1=BETA1_D)
                .minimize(D_loss, var_list=theta_D, global_step=global_step_d))

    lr_gen = tf.train.piecewise_constant(global_step_g, boundaries, values_G)
    G_solver = (tf.train.AdamOptimizer(learning_rate=lr_gen, beta1=BETA1_G)
                .minimize(G_loss, var_list=theta_G, global_step=global_step_g))

# summaries for visualization in Tensorboard
tf.summary.scalar("D_loss", D_loss)
tf.summary.scalar("G_loss", G_loss)
tf.summary.scalar("D data accuracy", tf.reduce_mean(D_enc))
tf.summary.scalar("D fake accuracy", tf.reduce_mean(1 - D_gen))
summary_op_scalar = tf.summary.merge_all()
summary_z_loc = summary(X_hat, title="generated_images_location")
summary_z_size = summary(X_hat, title="generated_images_size")

print("Initialize new session")
sess = tf.Session()
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
saver = tf.train.Saver(max_to_keep=1)
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

print("Start training")
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for iteration in range(1, MAX_ITER + 1):
    # train D and G
    if iteration % 1000 != 0:
        z_mb, Y_gen, box_generated = sample_generator_input(BATCH_SIZE, Z_DIM)
        _ = sess.run(D_solver, feed_dict={phase: 1, Y_: Y_gen, z: z_mb, bbox_: box_generated})
        for idx in range(args.g_train):
            z_mb, Y_gen, box_generated = sample_generator_input(BATCH_SIZE, Z_DIM)
            _ = sess.run(G_solver, feed_dict={phase: 1, Y_: Y_gen, z: z_mb, bbox_: box_generated}) #, Y_: Y_gen, z: z_mb, bbox_: box_generated

    # visualize training progress
    if iteration % 1000 == 0:
        z_mb, Y_gen, box_generated = sample_generator_input(BATCH_SIZE, Z_DIM)
        if args.metadata:
            # log metadata for train step of D
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([summary_op_scalar, D_solver],
                                  feed_dict={phase: 1, Y_: Y_gen, z: z_mb, bbox_: box_generated},
                                  options=run_options, run_metadata=run_metadata)
            summary_writer.add_run_metadata(run_metadata, 'step%03d' % iteration)
        else:
            summary, _ = sess.run([summary_op_scalar, D_solver],
                                  feed_dict={phase: 1, Y_: Y_gen, z: z_mb, bbox_: box_generated})

        for idx in range(args.g_train):
            z_mb, Y_gen, box_generated = sample_generator_input(BATCH_SIZE, Z_DIM)
            _ = sess.run(G_solver, feed_dict={phase: 1, Y_: Y_gen, z: z_mb, bbox_: box_generated})

        # visualize generated images via Tensorboard
        z_mb, Y_gen, box_generated = sample_generator_input(100, Z_DIM, sort_labels=True, sort_location=True)
        _summary_z_loc = sess.run(summary_z_loc, feed_dict={phase: 0, Y_: Y_gen, z: z_mb, bbox_: box_generated})

        z_mb, Y_gen, box_generated = sample_generator_input(100, Z_DIM, sort_labels=True, sort_location=True, sort_bbox_size=True)
        _summary_z_size = sess.run(summary_z_size, feed_dict={phase: 0, Y_: Y_gen, z: z_mb, bbox_: box_generated})

        summary_writer.add_summary(summary, iteration)
        summary_writer.add_summary(_summary_z_loc, iteration)
        summary_writer.add_summary(_summary_z_size, iteration)
        summary_writer.flush()

    # save model
    if iteration % 5000 == 0:
        if args.timeline:
            run_metadata = tf.RunMetadata()
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            with open(log_dir + '/timeline' + str(iteration) + '.ctf.json', 'w') as trace_file:
                trace_file.write(trace.generate_chrome_trace_format())
        snapshot_name = "iteration"
        fn = saver.save(sess, "{}/iteration.ckpt".format(log_dir), global_step=iteration)
        print("Model saved in file: {}".format(fn))
