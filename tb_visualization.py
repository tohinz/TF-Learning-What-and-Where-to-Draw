import tensorflow as tf


def pad_imgs(imgs, padding_color=(1), padding=(2,2)):
    """Add white border around images for easier visualization"""
    return tf.pad(imgs, [(0,0), padding, padding, (0,0)], mode="CONSTANT", constant_values=padding_color)


def summary(gen_imgs, title, rows=10, cols=10):
    """Create a rows x cols image of sub-images to be visualized via tensorboard"""
    gen_imgs = pad_imgs(gen_imgs)
    stacked_img = []
    for row in xrange(rows):
        row_img = []
        for col in xrange(cols):
            row_img.append(gen_imgs[(row * cols) + col, :, :, :])
        stacked_img.append(tf.concat(row_img, 1))
    imgs = tf.concat(stacked_img, 0)
    imgs = tf.expand_dims(imgs, 0)
    return tf.summary.image(title, imgs)
