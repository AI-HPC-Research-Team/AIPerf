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
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
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


def build_graph_from_json(ir_model_json):
    """build model from json representation
    """
    try:
        graph = json_to_graph(ir_model_json)
        logging.debug(graph.operation_history)
        model = graph.produce_tf_model()
        return model
    except Exception as E:
        print("#########:" + str(E))
        f = open('resnet50.json', 'r')
        a = json.load(f)
        RCV_CONFIG = json.dumps(a)
        f.close()
        graph = json_to_graph(RCV_CONFIG)
        model = graph.produce_tf_model()
        return model


def parse_rev_args(receive_msg, esargs):
    """ parse reveive msgs to global variable
    """
    global net
    global bs_explore
    global gpus
    # Model

    bs_explore = int(esargs['batch_size'])
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        net = build_graph_from_json(receive_msg)
        optimizer = SGD(lr=args.initial_lr, momentum=0.9, decay=1e-4)
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer, loss_scale=256)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=args.smooth_factor)

        denseList = []
        for index, layer in enumerate(net.layers):
            if isinstance(layer, tf.keras.layers.Dropout):
                net.layers[index].rate = esargs['dropout_rate']
            if isinstance(layer, tf.keras.layers.Dense):
                denseList.append(index)

        if len(denseList) == 0:
            thresh = 175
        elif len(denseList) == 1:
            thresh = denseList[0] + 1
        else:
            thresh = 175

        for index, layer in enumerate(net.layers):
            if index > thresh:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    net.layers[index].kernel_size = (esargs['kernel_size'], esargs['kernel_size'])

        # Compile the model
        net.compile(
            # loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
            loss=loss, optimizer=optimizer, metrics=["accuracy"]
        )


class SendMetrics(tf.keras.callbacks.Callback):
    """
    Keras callback to send metrics to NNI framework
    """

    def __init__(self, hp_path):
        super(SendMetrics, self).__init__()
        self.hp_path = hp_path
        self.best_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        """
        Run on end of each epoch
        """
        if logs is None:
            logs = dict()
        logger.debug(logs)
        with open(self.hp_path, 'r') as f:
            hp = json.load(f)
        hp['epoch'] = epoch + 1
        if logs['val_accuracy'] > self.best_acc:
            self.best_acc = logs['val_accuracy']
            hp['single_acc'] = logs['val_accuracy']
        hp['finish_date'] = time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(time.time()))
        with open(self.hp_path, 'w') as f:
            json.dump(hp, f)


def train_eval(esargs, RCV_CONFIG, seqid):
    """ train and eval the model
    """
    global net
    global best_acc
    global bs_explore
    global gpus
    global hp_path

    best_acc = 0
    parse_rev_args(RCV_CONFIG, esargs)
    # train procedure
    trial_id = nni.get_trial_id()
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
    checkpoint_dir = os.environ["HOME"] + "/nni/experiments/" + str(nni.get_experiment_id()) + "/checkpoint/" + str(
        nni.get_trial_id())
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_filepath = checkpoint_dir + "/weights." + "epoch." + str(run_epochs) + ".hdf5"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_freq='epoch',
        save_weights_only=True,
    )

    history = net.fit(
        ds_train,
        epochs=run_epochs,
        steps_per_epoch=Ntrain // bs_explore // gpus,
        validation_data=ds_val,
        validation_steps=Nvalidation // bs_explore // gpus,
        verbose=1,
        shuffle=False,
        callbacks=[SendMetrics(hp_path),
                   callback,
                   EarlyStopping(min_delta=0.001, patience=patience),
                   model_checkpoint_callback
                   ])

    # trial report final acc to tuner
    acc = 0
    acc_list = history.history['val_accuracy']
    for acc_n in acc_list:
        if float(acc_n) > acc:
            acc = float(acc_n)
    try:
        # predict acc
        if run_epochs >= 10 and run_epochs < 80:
            epoch_x = range(1, len(acc_list) + 1)
            pacc = utils.predict_acc(trial_id, epoch_x, acc_list, 90, True)
            best_acc = float(pacc)
    except Exception as E:
        print("Predict failed.")
    if acc > best_acc:
        best_acc = acc
    logger.debug("Final result is: %.3f", acc)
    return best_acc, history.epoch[-1]


if __name__ == "__main__":
    example_start_time = time.time()
    net = None
    args = get_args()
    try:
        experiment_path = os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id())
        lock = multiprocessing.Lock()
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        tmpstr = 'tcp://' + args.ip + ':800081'
        socket.connect(tmpstr)
        os.makedirs(experiment_path + "/trials/" + str(nni.get_trial_id()))

        get_next_parameter_start = time.time()
        nni.get_next_parameter(socket)
        get_next_parameter_end = time.time()

        while True:
            lock.acquire()
            with open(experiment_path + "/graph.txt", "a+") as f:
                f.seek(0)
                lines = f.readlines()
            lock.release()
            if lines:
                break

        if len(lines) > args.slave:
            x = random.randint(1, args.slave)
            json_and_id_str = lines[-x].replace("\n", "")
        else:
            json_and_id_str = lines[-1].replace("\n", "")

        with open(experiment_path + "/trials/" + str(nni.get_trial_id()) + "/output.log", "a+") as f:
            f.write("sequence_id=" + str(nni.get_sequence_id()) + "\n")
        json_and_id = dict((l.split('=') for l in json_and_id_str.split('+')))
        if str(json_and_id['history']) == "True":
            socket.send_pyobj({"type": "generated_parameter", "parameters": json_and_id['json_out'],
                               "father_id": int(json_and_id['father_id']), "parameter_id": int(nni.get_sequence_id())})
            message = socket.recv_pyobj()
        elif str(json_and_id['history']) == "False":
            socket.send_pyobj({"type": "generated_parameter"})
            message = socket.recv_pyobj()
        RCV_CONFIG = json_and_id['json_out']

        start_time = time.time()
        with open('search_space.json') as json_file:
            search_space = json.load(json_file)

        min_gpu_mem = utils.MinGpuMem()
        for index in range(len(search_space['batch_size']['_value'])):
            search_space['batch_size']['_value'][index] *= min_gpu_mem

        init_search_space_point = {"dropout_rate": 0.0, "kernel_size": 3, "batch_size": args.batch_size}

        if 'father_id' in json_and_id:
            json_father_id = int(json_and_id['father_id'])
            while True:
                if os.path.isfile(experiment_path + '/hyperparameter/' + str(json_father_id) + '.json'):
                    with open(experiment_path + '/hyperparameter/' + str(json_father_id) + '.json') as hp_json:
                        init_search_space_point = json.load(hp_json)
                    break
                elif json_father_id > 0:
                    json_father_id -= 1
                else:
                    break
        train_num = 0
        TPE = TPEtuner.HyperoptTuner('tpe')
        TPE.update_search_space(search_space)
        searched_space_point = {}
        start_date = time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(time.time()))

        current_json = json_and_id['json_out']
        current_hyperparameter = init_search_space_point
        if not os.path.isdir(experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id())):
            os.makedirs(experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id()))
        with open(experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id()) + '/model.json', 'w') as f:
            f.write(current_json)
        global hp_path
        hp_path = experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id()) + '/0.json'
        with open(hp_path, 'w') as f:
            json.dump(
                {'get_sequence_id': int(nni.get_sequence_id()), 'hyperparameter': current_hyperparameter, 'epoch': 0,
                 'single_acc': 0,
                 'train_time': 0, 'start_date': start_date}, f)

        pid = os.getpid()
        trial_log_path = os.environ["HOME"] + "/nni/experiments/" + str(nni.get_experiment_id()) + '/trials/' + str(
            nni.get_trial_id()) + '/trial.log'
        p = multiprocessing.Process(target=utils.trial_activity, args=(trial_log_path, pid,))
        p.daemon = True
        p.start()

        single_acc, current_ep = train_eval(init_search_space_point, RCV_CONFIG, int(nni.get_sequence_id()))
        print("HPO-" + str(train_num) + ",hyperparameters:" + str(init_search_space_point) + ",best_val_acc:" + str(
            single_acc))

        best_final = single_acc
        searched_space_point = init_search_space_point

        if int(nni.get_sequence_id()) > 3 * args.slave - 1:
            dict_first_data = init_search_space_point
            TPE.receive_trial_result(train_num, dict_first_data, single_acc)
            TPEearlystop = utils.EarlyStopping(patience=3, mode="max")

            for train_num in range(1, args.maxTPEsearchNum):
                params = TPE.generate_parameters(train_num)
                start_date = time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(time.time()))

                current_hyperparameter = params
                hp_path = experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id()) + '/' + str(
                    train_num) + '.json'
                with open(hp_path, 'w') as f:
                    json.dump(
                        {'get_sequence_id': int(nni.get_sequence_id()), 'hyperparameter': current_hyperparameter,
                         'epoch': 0, 'single_acc': 0,
                         'train_time': 0, 'start_date': start_date}, f)

                single_acc, current_ep = train_eval(params, RCV_CONFIG, int(nni.get_sequence_id()))
                print("HPO-" + str(train_num) + ",hyperparameters:" + str(params) + ",best_val_acc:" + str(single_acc))
                TPE.receive_trial_result(train_num, params, single_acc)

                if single_acc > best_final:
                    best_final = single_acc
                    searched_space_point = params
                if TPEearlystop.step(single_acc):
                    break

        nni.report_final_result(best_final, socket)
        if not os.path.isdir(experiment_path + '/hyperparameter'):
            os.makedirs(experiment_path + '/hyperparameter')
        with open(experiment_path + '/hyperparameter/' + str(nni.get_sequence_id()) + '.json',
                  'w') as hyperparameter_json:
            json.dump(searched_space_point, hyperparameter_json)

        end_time = time.time()

        with open(experiment_path + "/train_time", "w+") as f:
            f.write(str(end_time - start_time))

        with open(experiment_path + "/trials/" + str(nni.get_trial_id()) + "/output.log", "a+") as f:
            f.write("duration=" + str(time.time() - example_start_time) + "\n")
            f.write("best_acc=" + str(best_final) + "\n")

    except Exception as exception:
        logger.exception(exception)
        raise
    exit(0)
