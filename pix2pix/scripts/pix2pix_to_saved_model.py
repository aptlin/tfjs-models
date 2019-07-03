from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

parser = argparse.ArgumentParser()

parser.add_argument("--output_dir", required=True, help="where to put output files")

parser.add_argument("--seed", type=int)

parser.add_argument(
    "--separable_conv",
    default=False,
    action="store_true",
    help="use separable convolutions in the generator",
)

parser.add_argument(
    "--checkpoint",
    default=None,
    help="directory with checkpoint to resume training from or use for testing",
)

parser.add_argument(
    "--ngf",
    type=int,
    default=64,
    help="number of generator filters in first conv layer",
)

args = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

Examples = collections.namedtuple(
    "Examples", "paths, inputs, targets, count, steps_per_epoch"
)
Model = collections.namedtuple(
    "Model",
    "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, "
    + "gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train",
)


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(
        batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT"
    )
    return tf.layers.conv2d(
        padded_input,
        out_channels,
        kernel_size=4,
        strides=(stride, stride),
        padding="valid",
        kernel_initializer=tf.random_normal_initializer(0, 0.02),
    )


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if args.separable_conv:
        return tf.layers.separable_conv2d(
            batch_input,
            out_channels,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            depthwise_initializer=initializer,
            pointwise_initializer=initializer,
        )
    else:
        return tf.layers.conv2d(
            batch_input,
            out_channels,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            kernel_initializer=initializer,
        )


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if args.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(
            batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return tf.layers.separable_conv2d(
            resized_input,
            out_channels,
            kernel_size=4,
            strides=(1, 1),
            padding="same",
            depthwise_initializer=initializer,
            pointwise_initializer=initializer,
        )
    else:
        return tf.layers.conv2d_transpose(
            batch_input,
            out_channels,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            kernel_initializer=initializer,
        )


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(
        inputs,
        axis=3,
        epsilon=1e-5,
        momentum=0.1,
        training=True,
        gamma_initializer=tf.random_normal_initializer(1.0, 0.02),
    )


def check_image(image):
    assertion = tf.assert_equal(
        tf.shape(image)[-1], 3, message="image must have 3 color channels"
    )
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image


# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (
                ((srgb_pixels + 0.055) / 1.055) ** 2.4
            ) * exponential_mask
            rgb_to_xyz = tf.constant(
                [
                    #    X        Y          Z
                    [0.412453, 0.212671, 0.019334],  # R
                    [0.357580, 0.715160, 0.119193],  # G
                    [0.180423, 0.072169, 0.950227],  # B
                ]
            )
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(
                xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754]
            )

            epsilon = 6 / 29
            linear_mask = tf.cast(
                xyz_normalized_pixels <= (epsilon ** 3), dtype=tf.float32
            )
            exponential_mask = tf.cast(
                xyz_normalized_pixels > (epsilon ** 3), dtype=tf.float32
            )
            fxfyfz_pixels = (
                xyz_normalized_pixels / (3 * epsilon ** 2) + 4 / 29
            ) * linear_mask + (xyz_normalized_pixels ** (1 / 3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant(
                [
                    #  l       a       b
                    [0.0, 500.0, 0.0],  # fx
                    [116.0, -500.0, 200.0],  # fy
                    [0.0, 0.0, -200.0],  # fz
                ]
            )
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant(
                [-16.0, 0.0, 0.0]
            )

        return tf.reshape(lab_pixels, tf.shape(srgb))


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, args.ngf)
        layers.append(output)

    layer_specs = [
        args.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        args.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        args.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        args.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        args.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        args.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        args.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    layer_specs = [
        (
            args.ngf * 8,
            0.5,
        ),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (
            args.ngf * 8,
            0.5,
        ),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (
            args.ngf * 8,
            0.5,
        ),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (
            args.ngf * 8,
            0.0,
        ),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (
            args.ngf * 4,
            0.0,
        ),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (
            args.ngf * 2,
            0.0,
        ),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (
            args.ngf,
            0.0,
        ),  # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input_tensor = layers[-1]
            else:
                input_tensor = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input_tensor)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input_tensor = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input_tensor)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def main():
    if args.seed is None:
        args.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.checkpoint is None:
        raise Exception("Please provide the checkpoint directory.")

    # load some options from the checkpoint
    options = {"ngf"}
    with open(os.path.join(args.checkpoint, "options.json")) as f:
        for key, val in json.loads(f.read()).items():
            if key in options:
                setattr(args, key, val)

    input_tensor = tf.placeholder(tf.float32, shape=[CROP_SIZE, CROP_SIZE, 3])
    batch_input = tf.expand_dims(input_tensor, axis=0)

    with tf.variable_scope("generator"):
        output = deprocess(create_generator(preprocess(batch_input), 3))

    key = tf.placeholder(tf.string, shape=[1])
    inputs = {
        "key": tf.saved_model.utils.build_tensor_info(key),
        "input": tf.saved_model.utils.build_tensor_info(input_tensor),
    }
    outputs = {
        "key": tf.saved_model.utils.build_tensor_info(tf.identity(key)),
        "output": tf.saved_model.utils.build_tensor_info(output),
    }
    signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=outputs,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME,
    )
    init_op = tf.global_variables_initializer()
    restore_saver = tf.train.Saver()
    builder = tf.saved_model.builder.SavedModelBuilder(args.output_dir)

    with tf.Session() as sess:
        sess.run(init_op)
        checkpoint = tf.train.latest_checkpoint(args.checkpoint)
        restore_saver.restore(sess, checkpoint)
        builder.add_meta_graph_and_variables(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature
            },
            strip_default_attrs=True,
        )
        builder.save()


main()
