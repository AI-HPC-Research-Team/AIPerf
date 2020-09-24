# Copyright (c) Peng Cheng Laboratory
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import tensorflow as tf
import imagenet_preprocessing

def get_filenames(data_dir):
    files = os.listdir(data_dir)
    TF_files = []
    for i in files:
        TF_files.append(os.path.join(data_dir,str(i)))
    return TF_files


def process_record_dataset(dataset, is_training, batch_size, shuffle_buffer,
                           parse_record_fn, num_epochs=1, num_gpus=None,
                           examples_per_epoch=None, dtype=tf.float32):

    dataset = dataset.prefetch(buffer_size=batch_size)

    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer)
    dataset = dataset.repeat(num_epochs)

    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
        lambda value: parse_record_fn(value, is_training, dtype),
        batch_size=batch_size * num_gpus
        , num_parallel_calls=48))

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def _parse_example_proto(example_serialized):


    featdef = {
        'image/encoded': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
        'image/class/label': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),
    }

    example = tf.io.parse_single_example(example_serialized, featdef)
    im = example['image/encoded']
    label = tf.cast(example['image/class/label'] - 1, tf.int32)
    label = tf.one_hot(label, 1000)
    return im, label


def parse_record(raw_record, is_training, dtype=tf.float32):
    image_buffer, label = _parse_example_proto(raw_record)
    image_buffer = imagenet_preprocessing.preprocess_image(
        image_buffer=image_buffer,
        output_height=224,
        output_width=224,
        num_channels=3,
        is_training=is_training)
    image = tf.cast(image_buffer, dtype)
    return image, label
