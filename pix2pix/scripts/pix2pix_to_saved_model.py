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
    "--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"]
)
parser.add_argument(
    "--ngf",
    type=int,
    default=64,
    help="number of generator filters in first conv layer",
)
parser.add_argument(
    "--ndf",
    type=int,
    default=64,
    help="number of discriminator filters in first conv layer",
)
parser.add_argument(
    "--scale_size",
    type=int,
    default=286,
    help="scale images to this size before cropping to 256x256",
)
parser.add_argument(
    "--flip", dest="flip", action="store_true", help="flip images horizontally"
)
parser.add_argument(
    "--no_flip",
    dest="flip",
    action="store_false",
    help="don't flip images horizontally",
)
parser.set_defaults(flip=True)
parser.add_argument(
    "--lr", type=float, default=0.0002, help="initial learning rate for adam"
)
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument(
    "--l1_weight",
    type=float,
    default=100.0,
    help="weight on L1 term for generator gradient",
)
parser.add_argument(
    "--gan_weight",
    type=float,
    default=1.0,
    help="weight on GAN term for generator gradient",
)

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

Examples = collections.namedtuple(
    "Examples", "paths, inputs, targets, count, steps_per_epoch"
)
Model = collections.namedtuple(
    "Model",
    "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train",
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
    if a.separable_conv:
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
    if a.separable_conv:
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
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        a.ngf * 8,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
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
            a.ngf * 8,
            0.5,
        ),  # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (
            a.ngf * 8,
            0.5,
        ),  # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (
            a.ngf * 8,
            0.5,
        ),  # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (
            a.ngf * 8,
            0.0,
        ),  # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (
            a.ngf * 4,
            0.0,
        ),  # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (
            a.ngf * 2,
            0.0,
        ),  # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (
            a.ngf,
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
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2 ** 31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.checkpoint is None:
        raise Exception("checkpoint required for test mode")

    # load some options from the checkpoint
    options = {"which_direction", "ngf", "ndf", "lab_colorization"}
    with open(os.path.join(a.checkpoint, "options.json")) as f:
        for key, val in json.loads(f.read()).items():
            if key in options:
                print("loaded", key, "=", val)
                setattr(a, key, val)
    # disable these features in test mode
    a.scale_size = CROP_SIZE
    a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    # with open(os.path.join(a.output_dir, "options.json"), "w") as f:
    #     f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    # export the generator to a meta graph that can be imported later for standalone generation
    if a.lab_colorization:
        raise Exception("export not supported for lab_colorization")

    input = tf.placeholder(tf.string, shape=[1])
    input_data = tf.decode_base64(input[0])
    input_image = tf.image.decode_png(input_data)

    # remove alpha channel if present
    input_image = tf.cond(
        tf.equal(tf.shape(input_image)[2], 4),
        lambda: input_image[:, :, :3],
        lambda: input_image,
    )
    # convert grayscale to RGB
    input_image = tf.cond(
        tf.equal(tf.shape(input_image)[2], 1),
        lambda: tf.image.grayscale_to_rgb(input_image),
        lambda: input_image,
    )

    input_image = tf.image.convert_image_dtype(input_image, dtype=tf.float32)
    input_image.set_shape([CROP_SIZE, CROP_SIZE, 3])
    batch_input = tf.expand_dims(input_image, axis=0)

    with tf.variable_scope("generator"):
        batch_output = deprocess(create_generator(preprocess(batch_input), 3))

    output_image = tf.image.convert_image_dtype(batch_output, dtype=tf.uint8)[0]
    if a.output_filetype == "png":
        output_data = tf.image.encode_png(output_image)
    elif a.output_filetype == "jpeg":
        output_data = tf.image.encode_jpeg(output_image, quality=80)
    else:
        raise Exception("invalid filetype")
    output = tf.convert_to_tensor([tf.encode_base64(output_data)])

    key = tf.placeholder(tf.string, shape=[1])
    inputs = {"key": key.name, "input": input.name}
    tf.add_to_collection("inputs", json.dumps(inputs))
    outputs = {"key": tf.identity(key).name, "output": output.name}
    tf.add_to_collection("outputs", json.dumps(outputs))

    init_op = tf.global_variables_initializer()
    restore_saver = tf.train.Saver()
    # export_saver = tf.train.Saver()
    builder = tf.saved_model.builder.SavedModelBuilder(a.output_dir)
    with tf.Session() as sess:
        sess.run(init_op)
        print("loading model from checkpoint")
        checkpoint = tf.train.latest_checkpoint(a.checkpoint)
        restore_saver.restore(sess, checkpoint)
        print("exporting model")
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.SERVING], strip_default_attrs=True
        )
        builder.save()
        # tf.saved_model.simple_save(sess, a.output_dir, inputs=inputs, outputs=outputs)
        # export_saver.export_meta_graph(
        #     filename=os.path.join(a.output_dir, "export.meta")
        # )
        # export_saver.save(
        #     sess, os.path.join(a.output_dir, "export"), write_meta_graph=False
        # )
    return


main()
