# Copyright (c) Microsoft Corporation
# Copyright (c) Tsinghua University
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

import argparse
import logging
import sys
import multiprocessing 
import nni
from nni.networkmorphism_tuner.graph import json_to_graph
import json
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torchvision
from torch.utils.data.distributed import DistributedSampler

from distributed_utils import dist_init, average_gradients, DistModule

import utils
import time
import datetime
import zmq
from nni.env_vars import trial_env_vars
import json
import os
import random
import yaml

from hyperopt import fmin, tpe, hp
import  nni.hyperopt_tuner.hyperopt_tuner as TPEtuner

# set the logger format
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    filename="networkmorphism.log",
    filemode="a",
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
# pylint: disable=W0603
# set the logger format
logger = logging.getLogger("cifar10-network-morphism-pytorch")


def get_args():
    """ get args from command line
    """
    parser = argparse.ArgumentParser("cifar10")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer")
    parser.add_argument("--epochs", type=int, default=5, help="epoch limit")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
    parser.add_argument("--cutout_length", type=int, default=8, help="cutout length")
    parser.add_argument(
        "--model_path", type=str, default="./", help="Path to save the destination model"
    )
    parser.add_argument('--port', default='23456', type=str)
    parser.add_argument('-j', '--workers', default=2, type=int)
    parser.add_argument("--maxTPEsearchNum", type=int, default=20, help="maxTPEsearchNum")
    return parser.parse_args()


trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0.0
args = get_args()


def build_graph_from_json(ir_model_json):
    """build model from json representation
    """
    graph = json_to_graph(ir_model_json)
    logging.debug(graph.operation_history)
    model = graph.produce_torch_model()
    return model

def parse_rev_args(receive_msg):
    """ parse reveive msgs to global variable
    """
    global trainloader
    global testloader
    global trainsampler
    global testsampler
    global net
    global criterion
    global optimizer
    global rank, world_size

    # Loading Data
    if rank == 0:
        logger.debug("Preparing data..")

    transform_train, transform_test = utils.data_transforms_cifar10(args)

    dataPath = os.environ["HOME"] + "/mountdir/data/"
    trainset = torchvision.datasets.CIFAR10(
        root=dataPath, train=True, download=True, transform=transform_train
    )
    #
    # trainsampler = DistributedSampler(trainset)
    #
    # trainloader = torch.utils.data.DataLoader(
    #     trainset, batch_size=args.batch_size_per_gpu, shuffle=False, num_workers=args.workers,
    #     pin_memory=False, sampler=trainsampler
    # )

    


    testset = torchvision.datasets.CIFAR10(
        root=dataPath, train=False, download=True, transform=transform_test
    )

    testsampler = DistributedSampler(testset)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory = False, sampler = testsampler
    )
    if rank == 0:
        print("len(trainset)=" + str(len(trainset)))
        print("len(testset)=" + str(len(testset)))

    # Model
    if rank == 0:
        logger.debug("Building model..")
    net = build_graph_from_json(receive_msg)

    net = net.to(device)
    net = DistModule(net)
    criterion = nn.CrossEntropyLoss()

    # if args.optimizer == "SGD":
    #     optimizer = optim.SGD(
    #         net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    #     )
    # if args.optimizer == "Adadelta":
    #     optimizer = optim.Adadelta(net.parameters(), lr=args.learning_rate)
    # if args.optimizer == "Adagrad":
    #     optimizer = optim.Adagrad(net.parameters(), lr=args.learning_rate)
    # if args.optimizer == "Adam":
    #     optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
    # if args.optimizer == "Adamax":
    #     optimizer = optim.Adamax(net.parameters(), lr=args.learning_rate)
    # if args.optimizer == "RMSprop":
    #     optimizer = optim.RMSprop(net.parameters(), lr=args.learning_rate)

    cudnn.benchmark = True

    return 0


# Training
def train(epoch,op_explore):
    """ train model on each epoch in trainset
    """

    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    global rank, world_size

    if rank == 0:
        logger.debug("Epoch: %d", epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    optimizer = op_explore
    f11=open('/root/log','a+')
    f11.write('### ready to train \n')
    f11.close()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        f11=open('/root/log','a+')
        f11.write('### loop to train \n')
        f11.close()
        targets = targets.cuda(async=True)
        #inputs, targets = inputs.to(device), targets.to(device)
        input_var = torch.autograd.Variable(inputs.cuda())
        target_var = torch.autograd.Variable(targets)

        optimizer.zero_grad()
        outputs = net(input_var)
        loss = criterion(outputs, target_var) / world_size

        loss.backward()
        average_gradients(net)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.data.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #As the cost of all_reduce, we don't use all_reduce every batch to calculate acc."
        """
        if rank == 0:
            logger.debug(
                "Loss: %.3f | Acc: %.3f%% (%d/%d)",
                train_loss / (batch_idx + 1),
                100.0 * tmp_correct / tmp_total,
                tmp_correct,
                tmp_total,
            )
        """
    reduced_total = torch.Tensor([total])
    reduced_correct = torch.Tensor([correct])
    reduced_total = reduced_total.cuda()
    reduced_correct = reduced_correct.cuda()
    dist.all_reduce(reduced_total)
    dist.all_reduce(reduced_correct)

    tmp_total = int(reduced_total[0])
    tmp_correct = int(reduced_correct[0])
    acc = 100.0 * tmp_correct / tmp_total

    return acc

def test(epoch):
    """ eval model on each epoch in testset
    """
    global best_acc
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    global rank, world_size

    if rank == 0:
        logger.debug("Eval on epoch: %d", epoch)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets) / world_size

            test_loss += loss.item()
            _, predicted = outputs.data.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #As the cost of all_reduce, we don't use all_reduce every batch to calculate acc."
            """
            if rank == 0:
                logger.debug(
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)",
                    test_loss / (batch_idx + 1),
                    100.0 * tmp_correct / tmp_total,
                    tmp_correct,
                    tmp_total,
                )"""

    reduced_total = torch.Tensor([total])
    reduced_correct = torch.Tensor([correct])
    reduced_total = reduced_total.cuda()
    reduced_correct = reduced_correct.cuda()
    dist.all_reduce(reduced_total)
    dist.all_reduce(reduced_correct)

    tmp_total = int(reduced_total[0])
    tmp_correct = int(reduced_correct[0])
    acc = 100.0 * tmp_correct / tmp_total
    if acc > best_acc:
        best_acc = acc
    return acc, best_acc

acclist=[]
reslist=[]
def estimate(esargs):
    global best_acc
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    global rank
    #重置早停对象
    early_stop = utils.EarlyStopping(mode="max")
    global best_acc
    best_acc = 0
    lr_explore = esargs['learning_rate']
    bs_explore = int(esargs['batch_size'])
    global trainloader
    transform_train, transform_test = utils.data_transforms_cifar10(args)
    trainset = torchvision.datasets.CIFAR10(root= "/root/mountdir/data/", train=True, download=True, transform=transform_train)
    trainsampler = DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=bs_explore, shuffle=False, num_workers=args.workers,
        pin_memory=False, sampler=trainsampler
    )

    op = optim.SGD(net.parameters(), lr=lr_explore, momentum=0.9, weight_decay=5e-4)

    for ep in range(args.epochs):
        current_ep = ep + 1
        if rank==0:
            if os.popen("grep epoch " + experiment_path + "/trials/" + str(nni.get_trial_id()) + "/output.log").read():
                os.system("sed -i '/^epoch/cepoch=" + str(ep+1) + "' " + experiment_path + "/trials/" + str(nni.get_trial_id()) + "/output.log")
            else:
                os.system("sed -i '$a\\epoch=" + str(ep+1) + "' " +  experiment_path + "/trials/" + str(nni.get_trial_id()) + "/output.log")
        try:
            train_acc = train(ep,op)
        except Exception as exception:
            f11=open('/root/log','a+')
            f11.write('###### training is error \n')
            f11.write(str(exception)+"\n")
            f11.close()
            acclist.append(0)
            return 0,current_ep
        test_acc, best_acc = test(ep)
        logger.debug(test_acc)
        if early_stop.step(test_acc):
            break
    list = [best_acc,bs_explore,str(lr_explore)[0:7]]
    reslist.append(list)
    acclist.append(best_acc)
    return best_acc,current_ep

if __name__ == "__main__":
    args = get_args()
    rank, world_size = dist_init(args.port)
    if rank == 1:
        f11=open('/root/rank'+str(rank),'a+')
        f11.write('rank:'+str(rank)+"\n")
        f11.write("world_size:"+str(world_size)+"\n")
        f11.close()
    example_start_time = time.time()
    try:
        real_model_file = os.path.join("/root", "real_model.json")
        experiment_path = os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id())
        assert(args.workers % world_size == 0)
        args.workers = args.workers // world_size
        #real_model_file = os.path.join(trial_env_vars.NNI_SYS_DIR, "real_model.json")
        if rank == 0:    # only works for single node
            lock = multiprocessing.Lock()
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect("tcp://172.17.0.10:800081")
            # trial get next parameter from network morphism tuner
            #path=os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/trials/" + str(nni.get_trial_id())
            os.makedirs(experiment_path + "/trials/" + str(nni.get_trial_id()))

            get_next_parameter_start=time.time()
            nni.get_next_parameter(socket)
            get_next_parameter_end = time.time()

            while True: 
                lock.acquire()
                f1 = open(experiment_path + "/graph.txt","a+")
                f1.seek(0)
                lines = f1.readlines()
                f1.close()
                lock.release()
                if lines:
                    break
            json_and_id_str = lines[-1].replace("\n","") #逆序读取并记录,数据组成字典
            json_and_id = dict((l.split('=') for l in json_and_id_str.split('+')))
            if str(json_and_id['history']) == "True":
                socket.send_pyobj({"type": "generated_parameter", "parameters": json_and_id['json_out'], "father_id": int(json_and_id['father_id']), "parameter_id": int(nni.get_sequence_id())})
                f11=open('/root/log','a+')
                f11.write('histtory is True so \nsend parameters')
                f11.close()
                message = socket.recv_pyobj()
                f11=open('/root/log','a+')
                f11.write('recv message: '+ str(message)+'\n')
                f11.close()
            elif str(json_and_id['history']) == "False":
                socket.send_pyobj({"type": "generated_parameter"})
                f11=open('/root/log','a+')
                f11.write('history is false so \nsend generated_parameter\n')
                f11.close()
                message = socket.recv_pyobj()
                f11=open('/root/log','a+')
                f11.write('history is false so \nsend generated_parameter\n')
                f11.close()
        
            RCV_CONFIG = json_and_id['json_out']
            parse_rev_args(RCV_CONFIG)
            f11=open('/root/log','a+')
            f11.write("RCV_CONFIG:"+str(RCV_CONFIG)+"\n")
            f11.close()
            with open(real_model_file, "w") as f:
                json.dump(RCV_CONFIG, f)
            #logger.info(RCV_CONFIG)
        else:
            while not os.path.isfile(real_model_file):
                time.sleep(5)
            with open(real_model_file, "r") as f:
                RCV_CONFIG = json.load(f)

        if rank ==0:
            start_time = time.time()
            f = open(experiment_path + "/trials/" + str(nni.get_trial_id()) + "/output.log", "a+")
            f.write("sequence_id=" + str(nni.get_sequence_id()) + "\n")
            f.close()
            with open('search_space.json') as json_file:
                search_space = json.load(json_file)
            ## 根据father_id读取相应的超参json文件
            # 在起始时，不需要读取该件
            if 'father_id' in json_and_id:
                with open(experiment_path + '/hyperparameter/' + str(json_and_id['father_id']) + '.json') as hp_json:
                    init_search_space_point = json.load(hp_json)

        #临时测试数据，后期改进可从参数中获取
        init_search_space_point={"learning_rate":0.001,"batch_size":128}

        #初始化变量
        train_num = 0
        train_acc = 0.0
        best_acc = 0.0
        best_final = 0
        if rank ==0:
            # 使用hyperopt_tuner中的API完成TPE超参搜索
            TPE = TPEtuner.HyperoptTuner('tpe')
            TPE.update_search_space(search_space)    #输入space.json
            searched_space_point={}
            #执行第一次训练，获取训练时间，方便判断是否需要继续超参搜索
            train_time=time.time()
            start_date = time.strftime('%H.%M.%S',time.localtime(time.time()))
        #第一次训练
        f11=open('/root/log','a+')
        f11.write('rank:'+str(rank)+"\n")
        f11.write("#####11 run estimate: num 1   "+"\n")
        f11.close()
        single_acc,current_ep = estimate(init_search_space_point)
        f11=open('/root/log','a+')
        f11.write("#####22 run estimate:  "+str(single_acc)+"\n")
        f11.close()
        if rank == 0:
            single_train_time = time.time() - train_time
            best_final = single_acc
            searched_space_point = init_search_space_point  #需要写入本地文件，供其他进程读取

        #记录当前trial的模型以及每次搜索的超参以及对应的epoch 2/3
        current_json = json_and_id['json_out']
        current_hyperparameter = init_search_space_point
        if rank == 0:
            if not os.path.isdir(experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id())):
                os.makedirs(experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id()))
            with open(experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id()) + '/model.json', 'w') as f:
                f.write(current_json)
            with open(experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id()) + '/0.json', 'w') as f:
                json.dump({'hyperparameter':current_hyperparameter,'epoch':current_ep,'single_acc':single_acc,'train_time':single_train_time,'start_date':start_date},f)

        #从配置文件读取设备数
        #yml_f = open('/tmp/nni/experiments/' + str(nni.get_experiment_id() + '/trials/' + str(nni.get_trial_id()) + '/config.yml'))
        #yml = yaml.load(yml_f.read(), Loader = yaml.BaseLoader)
        gpuNum = int(os.popen('nvidia-smi -L|wc -l').read().strip())
        if int(nni.get_sequence_id()) > gpuNum:
       # if train_time > average_train_time:
            ##构造一个字典,先试试是否可行
            if rank == 0:
                dict_first_data = init_search_space_point
                TPE.receive_trial_result(train_num,dict_first_data,single_acc)
            ## 增加TPE search 的早停机制
            TPEearlystop=utils.EarlyStopping(patience=5,mode="max")
            real_TPEparams_file = os.path.join(trial_env_vars.NNI_SYS_DIR, "real_TPEparams")
            for train_num in range(1,args.maxTPEsearchNum):
                if rank == 0:
                    hy_train_start_time = time.time()
                    params = TPE.generate_parameters(train_num)
                    start_date = time.strftime('%H.%M.%S',time.localtime(time.time()))
                    with open(real_TPEparams_file, "w") as f:
                        json.dump(params, f)
                else:
                    while not os.path.isfile(real_TPEparams_file):
                        time.sleep(5)
                    with open(real_TPEparams_file, "r") as f:
                        params = json.load(f)

                single_acc,current_ep = estimate(params)
                if rank == 0:
                    TPE.receive_trial_result(train_num,params,single_acc)
                    hy_train_time = time.time() - hy_train_start_time
#                train_num = train_num+1
        #记录当前trial的模型以及每次搜索的超参以及对应的epoch 3/3
                    current_hyperparameter = params
                    with open(experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id()) + '/' + str(train_num) + '.json', 'w') as f:
                        json.dump({'hyperparameter':current_hyperparameter,'epoch':current_ep,'single_acc':single_acc,'train_time':hy_train_time,'start_date':start_date},f)

                    if single_acc>best_final:
                        best_final = single_acc
                        searched_space_point = params
                if TPEearlystop.step(single_acc):
                    break
        #存储搜索出效果最好的超参
        if rank == 0:
            if not os.path.isdir(experiment_path + '/hyperparameter'):
                os.makedirs(experiment_path + '/hyperparameter')
            with open(experiment_path + '/hyperparameter/' + str(nni.get_sequence_id()) + '.json','w') as hyperparameter_json:
                json.dump(searched_space_point,hyperparameter_json)

            end_time = time.time()

            f2 = open(experiment_path + "/c_time","w+")
            f2.write(str(end_time - start_time))
            f2.close()
            f = open(experiment_path + "/trials/" + str(nni.get_trial_id()) + "/output.log", "a+")
            f.write("get_next_parameter_time=" + str(get_next_parameter_end-get_next_parameter_start) + "\n")
            f.write("example_time=" + str(time.time() - example_start_time) + "\n")
            f.write("duration=" + str(time.time() - start_time) + "\n")
            f.write("best_acc=" + str(best_final) + "\n")
            f.close()

        # trial report best_acc to tuner
            nni.report_final_result(best_final,socket)

    except Exception as exception:
        logger.exception(exception)
        raise
