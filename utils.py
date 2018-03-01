import numpy as np
import tensorflow as tf

def log(x):
    return tf.log(x + 1e-8)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = tf.shape(x)
    y_shapes = y.get_shape()
    y = tf.reshape(y, (x_shapes[0], 1, 1, y_shapes[1]))
    y = y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[1]])

    return tf.concat([x , y], 3)


def tf_compute_transformation_matrix_inverse(bbox, shape=16., img_height=64):
    rel_factor = float(shape)/img_height
    x, y, w, h = rel_factor*bbox[0], rel_factor*bbox[1], rel_factor*bbox[2], rel_factor*bbox[3]

    scale_x = (shape / w)
    scale_y = (shape / h)

    t_x = (shape - 2 * x) / w - 1
    t_y = (shape - 2 * y) / h - 1

    line0 = tf.stack((scale_x, 0.0, t_x))
    line1 = tf.stack((0.0, scale_y, t_y))

    transformation_matrix = tf.concat((line0, line1), axis=0)
    transformation_matrix = tf.reshape(transformation_matrix, (2, 3))

    return transformation_matrix


def tf_compute_transformation_matrix(bbox, shape=16., img_height=64):
    rel_factor = float(shape)/img_height
    x, y, w, h = rel_factor*bbox[0], rel_factor*bbox[1], rel_factor*bbox[2], rel_factor*bbox[3]

    t_x = (x+0.5*w-0.5*shape)/(0.5*shape)
    t_y = (y+0.5*h-0.5*shape)/(0.5*shape)

    scale_x = (w / shape)
    scale_y = (h / shape)

    line0 = tf.stack((scale_x, 0.0, t_x))
    line1 = tf.stack((0.0, scale_y, t_y))
    transformation_matrix = tf.concat((line0, line1), axis=0)
    transformation_matrix = tf.reshape(transformation_matrix, (2, 3))

    return transformation_matrix


def sample_gen_data(batch_size, z_dim=100):
    _z = tf_sample_z(batch_size, z_dim)
    _Y = tf_sample_label(batch_size)
    _bbox = tf_sample_bbox(batch_size)
    return _z, _Y, _bbox


def sample_gen_data_position(batch_size, z_dim=100):
    _z = tf_sample_z(batch_size, z_dim)
    _Y = tf_sample_label(batch_size)
    _bbox = sample_bbox_sorted(batch_size)
    return _z, _Y, _bbox


def sample_gen_label(mb_size):
    labels = np.random.multinomial(1, 10*[0.1], size=mb_size)
    return labels


def sample_bbox(mb_size):
    pos_box_x = np.random.randint(low=0, high=44, size=(mb_size, 1))
    pos_box_y = np.random.randint(low=0, high=44, size=(mb_size, 1))

    coin = np.random.binomial(1, 0.5)
    if coin < 0.1:
        scale_box_x = np.random.randint(low=8, high=16, size=(mb_size, 1))
    elif 0.1 < coin < 0.8:
        scale_box_x = np.random.randint(low=12, high=18, size=(mb_size, 1))
    else:
        scale_box_x = np.random.randint(low=16, high=21, size=(mb_size, 1))
    scale_box_y = np.random.randint(low=18, high=21, size=(mb_size, 1))

    boxes = np.concatenate((pos_box_x, pos_box_y, scale_box_x, scale_box_y), axis=1)
    return boxes


def tf_sample_label(mb_size):
    labels = tf.random_uniform(shape=[mb_size], minval=0, maxval=10, dtype=tf.int32)
    labels = tf.one_hot(labels, 10)
    return labels


def tf_sample_bbox(mb_size):
    pos_box_x = tf.random_uniform(minval=0, maxval=44, shape=(mb_size, 1), dtype=tf.int32)
    pos_box_y = tf.random_uniform(minval=0, maxval=44, shape=(mb_size, 1), dtype=tf.int32)

    scale_box_x = tf.random_uniform(minval=8, maxval=21, shape=(mb_size, 1), dtype=tf.int32)
    scale_box_y = tf.random_uniform(minval=18, maxval=21, shape=(mb_size, 1), dtype=tf.int32)

    boxes = tf.concat((pos_box_x, pos_box_y, scale_box_x, scale_box_y), axis=1)
    return tf.cast(boxes, tf.float32)


def sample_bbox_sorted(mb_size):
    pos_box_x = np.zeros((mb_size, 1))
    for idx in range(100):
        pos_box_x[idx, 0] = (idx % 10 + 1) * 4

    pos_box_y = np.zeros((mb_size, 1))
    for idx in range(10):
        pos_box_y[idx * 10:idx * 10 + 10, 0] = (idx + 1) * 4

    scale_box_x = np.random.randint(low=16, high=21, size=(mb_size, 1))
    scale_box_y = np.random.randint(low=18, high=21, size=(mb_size, 1))

    boxes = np.concatenate((pos_box_x, pos_box_y, scale_box_x, scale_box_y), axis=1)
    return boxes


def sample_gen_label_sorted(mb_size, label_dim=10):
    labels = np.zeros((mb_size, label_dim))
    for idx in range(label_dim):
        labels[idx*label_dim:idx*label_dim+label_dim, idx] = 1
    return labels


def sample_generator_input(mb_size, n, sorted=False):
    _z = sample_z(mb_size, n)
    _Y = sample_gen_label(mb_size)
    if sorted:
        _bbox = sample_bbox_sorted(mb_size)
    else:
        _bbox = sample_bbox(mb_size)

    return _z, _Y, _bbox


def tf_sample_z(m, n):
    return tf.random_uniform(shape=[m, n], minval=-1., maxval=1.,)


def sample_z(m, n):
    """
    Sample noise z for generating new images.
    :param m: number of samples (rows)
    :param n: number of columns (size of z dimensionality)
    """
    return np.random.uniform(-1., 1., size=[m, n])