# Copyright 2019 Google Inc. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import argparse
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Model, Sequential


def build_generator():
    n_generator_filters = 32

    input = Input(shape=[256, 256, 3])
    layers = []
    x = Conv2D(
        filters=n_generator_filters,
        kernel_size=4,
        strides=(2, 2),
        padding="same",
        kernel_initializer="zeros",
        input_shape=[256, 256, 3],
    )(input)
    layers.append(x)

    layer_specs = [
        n_generator_filters * 2,
        n_generator_filters * 4,
        n_generator_filters * 8,
        n_generator_filters * 8,
        n_generator_filters * 8,
        n_generator_filters * 8,
        n_generator_filters * 8,
    ]

    for out_channels in layer_specs:
        x = LeakyReLU(alpha=0.2)(layers[-1])
        x = Conv2D(
            filters=out_channels,
            kernel_size=4,
            strides=(2, 2),
            padding="same",
            kernel_initializer="zeros",
            input_shape=[256, 256, 3],
        )(x)
        x = BatchNormalization(axis=3)(x, training=1)
        layers.append(x)

    layer_specs = [
        (n_generator_filters * 8, 0.5),
        (n_generator_filters * 8, 0.5),
        (n_generator_filters * 8, 0.5),
        (n_generator_filters * 8, 0.0),
        (n_generator_filters * 4, 0.0),
        (n_generator_filters * 2, 0.0),
        (n_generator_filters, 0.0),
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1

        if decoder_layer == 0:
            in_data = layers[-1]
        else:
            in_data = Concatenate(axis=3)([layers[-1], layers[skip_layer]])
        x = Activation("relu")(in_data)
        x = Conv2DTranspose(
            out_channels, kernel_size=4, strides=(2, 2), padding="same"
        )(x)
        x = BatchNormalization(axis=3)(x, training=1)

        if dropout > 0.0:
            x = Dropout(dropout)(x)
        layers.append(x)

    x = Concatenate(axis=3)([layers[-1], layers[0]])
    x = Activation("relu")(x)
    x = Conv2DTranspose(3, kernel_size=4, strides=(2, 2), padding="same")(x)
    output = Activation("tanh")(x)
    layers.append(output)

    return Model(inputs=input, outputs=output)


def convert_pix2pix(source_dir, target_dir, tmp_dirpath):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(source_dir + "/export.meta")
        saver.restore(sess, source_dir + "/export")
        idx = 0
        for var in tf.all_variables():
            out = sess.run(var)
            np.save(tmp_dirpath + "/" + str(idx) + ".npy", out)
            idx += 1
    tf.reset_default_graph()
    model = build_generator()

    weights = []
    for i in range(0, 88):
        name = tmp_dirpath + "/" + str(i) + ".npy"
        weights.append(np.load(name))

    idx = 0
    for layer in model.layers[1:]:
        if "conv2d" in layer.name:
            W = weights[idx]
            b = weights[idx + 1]
            layer.set_weights([W, b])
            idx += 2
        elif "batch" in layer.name:
            g = weights[idx]
            b = weights[idx + 1]
            m = weights[idx + 2]
            v = weights[idx + 3]
            layer.set_weights([g, b, m, v])
            idx += 4
        else:
            continue

    model.save(args.out + "/keras.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "This script converts Pix2Pix pretrained models, "
            "available from https://github.com/affinelayer/pix2pix-tensorflow, "
            "to tf.SavedModel."
        )
    )
    parser.add_argument(
        "--source", help="The folder containing the TensorFlow checkpoints"
    )
    parser.add_argument("--target", help="The")

    args = parser.parse_args()
    tmp_dirpath = tempfile.mkdtemp()
    convert_pix2pix(args.source, args.target, tmp_dirpath)
    shutil.rmtree(tmp_dirpath)

