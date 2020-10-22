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
import time
import zmq
import random
import json
import nni
import nni.hyperopt_tuner.hyperopt_tuner as TPEtuner
import multiprocessing
from multiprocessing import Process, Queue, RLock

import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from mindspore import context as mds_context
from mindspore import Tensor
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model, ParallelMode
from mindspore.train.callback import Callback, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.communication.management import init
from mindspore.common import set_seed
import utils
from dataset import create_dataset2 as create_dataset
from CrossEntropySmooth import CrossEntropySmooth
from lr_generator import get_lr, warmup_cosine_annealing_lr
from metric import DistAccuracy, ClassifyCorrectCell

# set the logger format
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    filename="networkmorphism.log",
    filemode="a",
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
# set the logger format
logger = logging.getLogger("Imagenet-network-morphism-tfkeras")


# imagenet2012
Ntrain = 1281167
Nvalidation = 50000
acclist = []
reslist = []
shuffle_buffer = 1024
examples_per_epoch = shuffle_buffer
TENSORBOARD_DIR = os.environ["NNI_OUTPUT_DIR"]


def get_args():
    """ get args from command line
    """
    parser = argparse.ArgumentParser("imagenet")
    parser.add_argument("--ip", type=str, default='127.0.0.1', help="ip address")
    parser.add_argument("--train_data_dir", type=str, default=None, help="tain data directory")
    parser.add_argument("--val_data_dir", type=str, default=None, help="val data directory")
    parser.add_argument("--slave", type=int, default=2, help="trial concurrency")
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--epochs", type=int, default=60, help="epoch limit")
    parser.add_argument("--initial_lr", type=float, default=1e-1, help="init learning rate")
    parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
    parser.add_argument("--maxTPEsearchNum", type=int, default=2, help="max TPE search number")
    parser.add_argument("--smooth_factor", type=float, default=0.1, help="max TPE search number")
    parser.add_argument("--warmup_1", type=int, default=15, help="epoch of first warm up round")
    parser.add_argument("--warmup_2", type=int, default=30, help="epoch of second warm up round")
    parser.add_argument("--warmup_3", type=int, default=45, help="epoch of third warm up round")
    return parser.parse_args()


def build_graph_from_json(model_json, param_json=''):
    """
    build model from json representation
    """
    from networkmorphism_tuner.graph import json_to_graph
    from networkmorphism_tuner.ProcessJson import ModifyJson

    if param_json != '':
        modify = ModifyJson(model_json, param_json)
        modify_json = modify.modify_hyper_parameters()
    else:
        modify_json = model_json

    graph = json_to_graph(modify_json)
    logging.debug(graph.operation_history)
    model = graph.produce_MindSpore_model()
    return model


def write_result_to_json(hp_path, epoch_size, acc):
    '''
    
    '''
    with open(hp_path, 'r') as f:
        hp = json.load(f)

    hp['epoch'] = epoch_size
    if acc > float(hp['single_acc']):
        hp['single_acc'] = acc
    hp['finish_date'] = time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(time.time()))

    with open(hp_path, 'w') as f:
        json.dump(hp, f)


class Accuracy(Callback):
    def __init__(self, model, dataset_val, device_id, epoch_size, data_size, ms_lock):
        super(Accuracy, self).__init__()
        self.model = model
        self.dataset_val = dataset_val
        self.device_id = device_id
        self.epoch_size = epoch_size
        self.data_size = data_size
        self.ms_lock = ms_lock

    def epoch_begin(self, run_context):
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch_num = cb_params.cur_epoch_num
        loss = cb_params.net_outputs
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / self.data_size

        self.ms_lock.acquire()
        print("[Device {}] Epoch {}/{}, train time: {:5.3f}, per step time: {:5.3f}, loss: {}".format(
            self.device_id, epoch_num, self.epoch_size, epoch_mseconds, per_step_mseconds, loss), flush=True)
        self.ms_lock.release()

        cur_time = time.time()
        acc = self.model.eval(self.dataset_val)['acc']
        val_time = int(time.time() - cur_time)

        self.ms_lock.acquire()
        print("[Device {}] Epoch {}/{}, EvalAcc:{}, EvalTime {}s".format(
                self.device_id, epoch_num, self.epoch_size, acc, val_time), flush=True)
        self.ms_lock.release()


def mds_train_eval(q, hyper_params, receive_config, dataset_path_train, dataset_path_val, epoch_size, batch_size, hp_path, device_id, device_num, enable_hccl, ms_lock):
    '''
    net:
    dataset_path_train:
    dataset_path_val:
    epoch_size:
    batch_size:
    hp_path:

    '''
    set_seed(1)
    target = 'Ascend'

    import socket as sck
    kernel_meta_file = sck.gethostname() + '_' + str(device_id)
    if os.path.exists(kernel_meta_file):
        os.system("rm -rf " + str(kernel_meta_file))
    os.system("mkdir " + str(kernel_meta_file))
    os.chdir(str(kernel_meta_file))
    ms_lock.acquire()
    print('++++  container: {}'.format(sck.gethostname()))
    ms_lock.release()
    # init context
    mds_context.set_context(mode=mds_context.GRAPH_MODE, device_target=target, save_graphs=False)
    mds_context.set_context(device_id=device_id)
    mds_context.set_context(max_call_depth=2000)

    os.environ['RANK_TABLE_FILE'] = os.environ['RANK_TABLE']
    os.environ['RANK_ID'] = str(device_id)
    os.environ['RANK_SIZE'] = str(device_num)
    os.environ['DEVICE_ID'] = str(device_id)
    os.environ['DEVICE_NUM'] = str(device_num)

    if enable_hccl:
        mds_context.set_context(device_id=device_id, enable_auto_mixed_precision=True)
        mds_context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        auto_parallel_context().set_all_reduce_fusion_split_indices([85, 160])
        init()

    eval_batch_size = 32
    # create dataset
    dataset_train = create_dataset(dataset_path=dataset_path_train, do_train=True, repeat_num=1, batch_size=batch_size, target=target)
    step_size = dataset_train.get_dataset_size()
    dataset_val = create_dataset(dataset_path=dataset_path_val, do_train=False, repeat_num=1, batch_size=eval_batch_size, target=target)

    # build network
    net = build_graph_from_json(receive_config, hyper_params)

    # evaluation network
    dist_eval_network = ClassifyCorrectCell(net)

    # init weight
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Conv2d):
            cell.weight.set_data(weight_init.initializer(weight_init.XavierUniform(),
                                                        cell.weight.shape,
                                                        cell.weight.dtype))
        if isinstance(cell, nn.Dense):
            cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                        cell.weight.shape,
                                                        cell.weight.dtype))

    # init lr
    lr = get_lr(lr_init=0.0, lr_end=0.0, lr_max=0.8, warmup_epochs=0,
                total_epochs=epoch_size, steps_per_epoch=step_size, lr_decay_mode='linear')
    lr = Tensor(lr)

    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': 1e-4},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    opt = Momentum(group_params, lr, 0.9, loss_scale=1024)

    # define loss, model
    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=0.1, num_classes=1001)
    loss_scale = FixedLossScaleManager(loss_scale=1024, drop_overflow_update=False)

    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, amp_level="O2", keep_batchnorm_fp32=False,
                  metrics={'acc': DistAccuracy(batch_size=eval_batch_size, device_num=device_num)},
                  eval_network=dist_eval_network)

    # define callbacks
    acc_cb = Accuracy(model, dataset_val, device_id, epoch_size, step_size, ms_lock)
    cb = [acc_cb]

     # train model
    model.train(epoch_size, dataset_train, callbacks=cb, dataset_sink_mode=True)

    # evaluation model
    acc = model.eval(dataset_val)['acc']
    q.put({'acc': acc})


def train_eval_distribute(hyper_params, receive_config, trial_id, hp_path):
    '''
    hyper_param:
    receive_config:
    trial_id:
    hp_path:
    '''

    # caculate epoch
    loopnum = trial_id // args.slave
    patience = min(int(6 + (2 * loopnum)), 20)
    if loopnum == 0:
        run_epochs = int(args.warmup_1)
    elif loopnum == 1:
        run_epochs = int(args.warmup_2)
    elif loopnum == 2:
        run_epochs = int(args.warmup_3)
    else:
        run_epochs = int(args.epochs)

    # build queue to save result for each process
    q = Queue()

    # base parameters
    device_num = int(os.getenv("NPU_NUM"))
    epoch_size = run_epochs
    batch_size = (int(hyper_params['batch_size']) // 16) * 16
    enable_hccl = True
    process = []
    ms_lock = RLock()
    # distribute training ...
    for i in range(device_num):
        device_id = i
        process.append(Process(target=mds_train_eval,
                               args=(q, hyper_params, receive_config, args.train_data_dir, args.val_data_dir, 
                                   epoch_size, batch_size, hp_path, device_id, device_num, enable_hccl, ms_lock)))
    for i in range(device_num):
        process[i].start()

    print("Waiting for all subprocesses done...")

    for i in range(device_num):
        process[i].join()

    # release resource
    process.clear()

    # get acc result
    acc = 0
    for i in range(device_num):
        try:
            output = q.get(block=False)
        except Queue.Empty:
            print('MindSpore train failed')
        acc += output['acc']
    acc = acc / device_num

   # 获取每一个 train 的起始终止位置（一个 trial.log 可能包含两次 train 过程）
    lines = []
    with open('/root/nni/experiments/{}/trials/{}/trial.log'.format(nni.get_experiment_id(), nni.get_trial_id()), 'r') as f:
        for index, line in enumerate(f.readlines()):
            lines.append(line)

    from utils import get_one_train_info, predict_acc
    best_acc = 0
    acc_list = get_one_train_info(lines, 0, -1)['eval_acc']
    try:
        if run_epochs >= 10 and run_epochs < 80:
            epoch_x = range(1, len(acc_list) + 1)
            pacc = predict_acc('', epoch_x, acc_list, 90, False)
            best_acc = float(pacc)
    except Exception as E:
        print("Predict failed.")
    if acc > best_acc:
        best_acc = acc

    print("End training...")
    write_result_to_json(hp_path, epoch_size, best_acc)
    logger.debug("Final result is: %.3f", best_acc)
 
    return best_acc, epoch_size


if __name__ == "__main__":
    example_start_time = time.time()
    args = get_args()
    try:
        experiment_path = os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id())

        # 开启模型生成和模型训练的异步执行
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

        # 增加 graph 随机读取扰动
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
            json.dump({'hyperparameter': current_hyperparameter, 
                       'epoch': 0, 
                       'single_acc': 0,
                       'train_time': 0, 
                       'start_date': start_date}, f)

        single_acc, current_ep = train_eval_distribute(init_search_space_point, RCV_CONFIG, int(nni.get_sequence_id()), hp_path)
        print("HPO-" + str(train_num) + ",hyperparameters:" + str(init_search_space_point) + ",best_val_acc:" + str(single_acc))

        # single_train_time = time.time() - train_time
        best_final = single_acc
        searched_space_point = init_search_space_point

        if int(nni.get_sequence_id()) > 3 * args.slave - 1:
            time.sleep(10)

            dict_first_data = init_search_space_point
            TPE.receive_trial_result(train_num, dict_first_data, single_acc)
            TPEearlystop = utils.EarlyStopping(patience=3, mode="max")

            for train_num in range(1, args.maxTPEsearchNum):
                params = TPE.generate_parameters(train_num)
                start_date = time.strftime('%m/%d/%Y, %H:%M:%S', time.localtime(time.time()))

                current_hyperparameter = params
                hp_path = experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id()) + '/' + str(train_num) + '.json'
                with open(hp_path, 'w') as f:
                    json.dump({'hyperparameter': current_hyperparameter, 
                               'epoch': 0, 
                               'single_acc': 0,
                               'train_time': 0, 
                               'start_date': start_date}, f)

                single_acc, current_ep = train_eval_distribute(params, RCV_CONFIG, int(nni.get_sequence_id()), hp_path)
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
        with open(experiment_path + '/hyperparameter/' + str(nni.get_sequence_id()) + '.json', 'w') as hyperparameter_json:
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
