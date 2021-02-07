# Copyright (c) Microsoft Corporation
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

# -*-coding:utf-8-*-

import argparse
import logging
import os
import json
import time
import zmq
import random
import numpy as np
import multiprocessing

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD

import nni
from nni.networkmorphism_tuner.graph import json_to_graph
import nni.hyperopt_tuner.hyperopt_tuner as TPEtuner

import utils
import imagenet_preprocessing
import dataset as ds

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    filename="networkmorphism.log",
    filemode="a",
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
logger = logging.getLogger("Imagenet-network-morphism-tfkeras")

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

# imagenet2012
Ntrain = 1281167
Nvalidation = 50000
shuffle_buffer = 1024
examples_per_epoch = shuffle_buffer
tf.config.optimizer.set_jit(True)

def get_args():
    """ get args from command line
    """
    parser = argparse.ArgumentParser("imagenet")
    parser.add_argument("--ip", type=str, default='127.0.0.1', help="ip address")
    parser.add_argument("--train_data_dir", type=str, default=None, help="tain data directory")
    parser.add_argument("--val_data_dir", type=str, default=None, help="val data directory")
    parser.add_argument("--slave", type=int, default=2, help="trial concurrency")
    parser.add_argument("--batch_size", type=int, default=448, help="batch size")
    parser.add_argument("--warmup_1", type=int, default=15, help="epoch of first warm up round")
    parser.add_argument("--warmup_2", type=int, default=30, help="epoch of second warm up round")
    parser.add_argument("--warmup_3", type=int, default=45, help="epoch of third warm up round")
    parser.add_argument("--epochs", type=int, default=60, help="epoch limit")
    parser.add_argument("--initial_lr", type=float, default=1e-1, help="init learning rate")
    parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
    parser.add_argument("--maxTPEsearchNum", type=int, default=2, help="max TPE search number")
    parser.add_argument("--smooth_factor", type=float, default=0.1, help="max TPE search number")
    parser.add_argument("--num_parallel_calls", type=int, default=48, help="number of parallel call during data loading")
    return parser.parse_args()


def build_graph_from_json():
    """build model from json representation
    """
    f = open('resnet50.json', 'r')
    a = json.load(f)
    RCV_CONFIG = json.dumps(a)
    f.close()
    graph = json_to_graph(RCV_CONFIG)
    model = graph.produce_tf_model()
    return model


def parse_rev_args(args):
    """ parse reveive msgs to global variable
    """
    global net
    global bs_explore
    global gpus
    # Model

    bs_explore = args.batch_size
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        net = build_graph_from_json()
        optimizer = SGD(lr=args.initial_lr, momentum=0.9, decay=1e-4)
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer, loss_scale=256)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.smooth_factor)

        # Compile the model
        net.compile(
            # loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
            loss=loss, optimizer=optimizer, metrics=["accuracy"]
        )


# class SendMetrics(tf.keras.callbacks.Callback):
#     """
#     Keras callback to send metrics to NNI framework
#     """

#     def __init__(self, hp_path):
#         super(SendMetrics, self).__init__()
#         self.hp_path = hp_path
#         self.best_acc = 0

#     def on_epoch_end(self, epoch, logs=None):
#         """
#         Run on end of each epoch
#         """
#         if logs is None:
#             logs = dict()
#         logger.debug(logs)
#         with open(self.hp_path, 'r') as f:
#             hp = json.load(f)
#         hp['epoch'] = epoch + 1
#         if logs['val_accuracy'] > self.best_acc:
#             self.best_acc = logs['val_accuracy']
#             hp['single_acc'] = logs['val_accuracy']
#         hp['finish_date'] = time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(time.time()))
#         with open(self.hp_path, 'w') as f:
#             json.dump(hp, f)


def train_eval(args):
    """ train and eval the model
    """
    global net
    global best_acc
    global bs_explore
    global gpus
    global hp_path

    best_acc = 0
    parse_rev_args(args)
    # train procedure
    available_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    gpus = len(available_devices.split(","))

    is_training = True
    filenames = ds.get_filenames(args.train_data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    ds_train = ds.process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=bs_explore,
        shuffle_buffer=shuffle_buffer,
        parse_record_fn=ds.parse_record,
        num_epochs=args.epochs,
        npc=args.num_parallel_calls,
        num_gpus=gpus,
        examples_per_epoch=examples_per_epoch if is_training else None,
        dtype=tf.float32
    )

    is_training = False
    filenames = ds.get_filenames(args.val_data_dir)
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    ds_val = ds.process_record_dataset(
        dataset=dataset,
        is_training=is_training,
        batch_size=bs_explore,
        shuffle_buffer=shuffle_buffer,
        parse_record_fn=ds.parse_record,
        num_epochs=args.epochs,
        npc=args.num_parallel_calls,
        num_gpus=gpus,
        examples_per_epoch=None,
        dtype=tf.float32
    )

    # run epochs and patience
    loopnum = seqid // args.slave
    patience = min(int(6 + (2 * loopnum)), 20)
    if loopnum == 0:
        run_epochs = int(args.warmup_1)
    elif loopnum == 1:
        run_epochs = int(args.warmup_2)
    elif loopnum == 2:
        run_epochs = int(args.warmup_3)
    else:
        run_epochs = int(args.epochs)
    
    # if loopnum < 4:
    #     patience = int(8 + (2 * loopnum))
    #     run_epochs = int(10 + (20 * loopnum))
    # else:
    #     patience = 16
    #     run_epochs = args.epochs

    # lr strategy

    def scheduler2(epoch):
        lr_max = args.initial_lr
        total_epochs = args.epochs
        lr_each_epoch = lr_max - lr_max * epoch / total_epochs
        return lr_each_epoch

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler2)

    # save weights
    # checkpoint_dir = os.environ["HOME"] + "/nni/experiments/" + str(nni.get_experiment_id()) + "/checkpoint/" + str(
    #     nni.get_trial_id())
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    # checkpoint_filepath = checkpoint_dir + "/weights." + "epoch." + str(run_epochs) + ".hdf5"
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #     filepath=checkpoint_filepath,
    #     monitor='val_accuracy',
    #     mode='max',
    #     save_best_only=True,
    #     save_freq='epoch',
    #     save_weights_only=True,
    # )
    x_train = np.random.rand(10000,224,224,3)
    y_train = 0 * np.random.rand(10000,1000)
    x_test = np.random.rand(1000,224,224,3)
    y_test = 0 * np.random.rand(1000,1000)

    history = net.fit(x_train, y_train, batch_size=448, epochs=10, validation_data=(x_test, y_test), shuffle=True)

    history = net.fit(
        ds_train,
        epochs=args.epochs,
        steps_per_epoch=Ntrain // bs_explore // gpus,
        validation_data=ds_val,
        validation_steps=Nvalidation // bs_explore // gpus,
        verbose=1,
        shuffle=False,
        callbacks=[#SendMetrics(hp_path),
                   callback,
                   #EarlyStopping(min_delta=0.001, patience=patience),
                   #model_checkpoint_callback
                   ])


if __name__ == "__main__":
    example_start_time = time.time()
    net = None
    args = get_args()

    train_eval(args)
    

from PIL import Image

count = 0
x_train = np.zeros([5000, 224, 224, 3], dtype='float32')
y_train = np.ones([5000, 1000], dtype='float32')
dirpath = '/root/'
for filenames in os.listdir(dirpath):
    if count < 5000:
        filepath = os.path.join(dirpath, filenames)
        img_PIL = Image.open(filepath)
        img_PIL = np.array(img_PIL)
        img_PIL = img_PIL.resize((224, 224)) 
        x_train[count] = img_PIL
        count += 1

