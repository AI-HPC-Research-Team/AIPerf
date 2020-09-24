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
import os

import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, TensorBoard
from keras.datasets import cifar10
from keras.optimizers import SGD, Adadelta, Adagrad, Adam, Adamax, RMSprop
from keras.utils import multi_gpu_model, to_categorical
import keras.backend.tensorflow_backend as KTF

import multiprocessing

import nni
from nni.networkmorphism_tuner.graph import json_to_graph
import  nni.hyperopt_tuner.hyperopt_tuner as TPEtuner
import json
import utils

import time
import datetime
import zmq
import os
import random
import yaml
from distributed_utils import dist_init, average_gradients, DistModule
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
logger = logging.getLogger("Cifar10-network-morphism-keras")


# restrict gpu usage background
config = tf.ConfigProto()
# pylint: disable=E1101,W0603
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

KTF.set_session(sess)


def get_args():
    """ get args from command line
    """
    parser = argparse.ArgumentParser("cifar10")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--optimizer", type=str, default="SGD", help="optimizer")
    parser.add_argument("--epochs", type=int, default=2, help="epoch limit")
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="learning rate"
    )
    parser.add_argument("--maxTPEsearchNum", type=int, default=2, help="max TPE search number")
    parser.add_argument('--port', default='23456', type=str)
    parser.add_argument('-j', '--workers', default=2, type=int)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="weight decay of the learning rate",
    )
    return parser.parse_args()


trainloader = None
testloader = None
net = None
args = get_args()
TENSORBOARD_DIR = os.environ["NNI_OUTPUT_DIR"]


def build_graph_from_json(ir_model_json):
    """build model from json representation
    """
    graph = json_to_graph(ir_model_json)
    logging.debug(graph.operation_history)
    model = graph.produce_keras_model()
    return model


def parse_rev_args(receive_msg):
    """ parse reveive msgs to global variable
    """
    global trainloader
    global testloader
    global net
    global criterion
    global rank,world_size
    # Loading Data
    if rank == 0:
        logger.debug("Preparing data..")

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255.0
    x_test /= 255.0
    trainloader = (x_train, y_train)
    testloader = (x_test, y_test)
    ###################################################################
    #需要补充keras对于并行数据的处理
    ###################################################################

    # Model
    if rank == 0:
        logger.debug("Building model..")
    net = build_graph_from_json(receive_msg)
    # parallel model
    try:
        available_devices = os.environ["CUDA_VISIBLE_DEVICES"]
        gpus = len(available_devices.split(","))
        f11=open("/root/log","a+")
        f11.write("######GPU:"+str(gpus)+"\n")
        f11.close()
        if gpus > 1:
            net = multi_gpu_model(net, gpus)
    except KeyError:
        logger.debug("parallel model not support in this config settings")

    return 0


class SendMetrics(keras.callbacks.Callback):
    """
    Keras callback to send metrics to NNI framework
    """

    def on_epoch_end(self, epoch, logs=None):
        """
        Run on end of each epoch
        """
        if logs is None:
            logs = dict()
        logger.debug(logs)
        #nni.report_intermediate_result(logs["val_accuracy"])

class Epoch_num_record(keras.callbacks.Callback):
    """
    Keras callback to send metrics to NNI framework
    """

    def __init__(self, experiment_path, trial_id):
        super(Epoch_num_record, self).__init__()
        self.experiment_path = experiment_path
        self.trial_id = trial_id

    def on_epoch_begin(self, epoch, logs=None):
        if os.popen("grep epoch " + self.experiment_path + "/trials/" + str(self.trial_id) + "/output.log").read():
            os.system("sed -i '/^epoch/cepoch=" + str(epoch+1) + "' " + self.experiment_path + "/trials/" + str(self.trial_id) + "/output.log")
        else:
            os.system("sed -i '$a\\epoch=" + str(epoch+1) + "' " +  self.experiment_path + "/trials/" + str(self.trial_id) + "/output.log")


acclist=[]
reslist=[]
# Training
def train_eval(esargs):
    """ train and eval the model
    """

    global trainloader
    global testloader
    global net
    global best_acc
    global rank

    best_acc = 0
    lr_explore = esargs['learning_rate']
    bs_explore = int(esargs['batch_size'])
    if args.optimizer == "SGD":
        optimizer = SGD(lr=lr_explore, momentum=0, decay=args.weight_decay)
    elif args.optimizer == "Adadelta":
        optimizer = Adadelta(lr=lr_explore, decay=args.weight_decay)
    elif args.optimizer == "Adagrad":
        optimizer = Adagrad(lr=lr_explore, decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = Adam(lr=lr_explore, decay=args.weight_decay)
    elif args.optimizer == "Adamax":
        optimizer = Adamax(lr=lr_explore, decay=args.weight_decay)
    elif args.optimizer == "RMSprop":
        optimizer = RMSprop(lr=lr_explore, decay=args.weight_decay)
    else:
        logger.debug("Input A Wrong optimizer")

    # Compile the model
    net.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    (x_train, y_train) = trainloader
    (x_test, y_test) = testloader

    # train procedure
    #使用callback函数记录epoch信息
    trial_id=nni.get_trial_id()
    f11=open("/root/keras_trace"+str(rank),"a+")
    f11.write("rank-"+str(rank)+str(trial_id)+"\n")
    f11.close()
    available_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    gpus = len(available_devices.split(","))
    #需要打印看看GPU个数
    history = net.fit(
        x=x_train,
        y=y_train,
        batch_size=bs_explore*gpus,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        shuffle=True,
        callbacks=[
            SendMetrics(),
            Epoch_num_record(experiment_path,trial_id),
            EarlyStopping(min_delta=0.001, patience=10),
            TensorBoard(log_dir=TENSORBOARD_DIR),
        ],
    )

    # trial report final acc to tuner
    if rank ==0:
        _, acc = net.evaluate(x_test, y_test)
    #记录超参搜索期间产生的最优acc
        f11=open("/root/log","a+")
        f11.write("######acc:"+str(acc)+"\n")
        f11.close()
        if acc > best_acc:
            best_acc = acc

        logger.debug("Final result is: %.3f", acc)
        list = [best_acc, bs_explore, str(lr_explore)[0:7]]
        reslist.append(list)
        acclist.append(best_acc)
    return best_acc,history.epoch[-1]

    #需要将该报告函数放到main中，用于报告最后超参搜索完成的acc
    #nni.report_final_result(acc)







if __name__ == "__main__":
    #nni.init_trial()
    #rank, world_size = dist_init(args.port)
    rank=0
    f11=open("/root/keras_trace"+str(rank),"a+")
    f11.write("rank-"+str(rank)+"\n")
    f11.close()
    example_start_time = time.time()
    try:
        real_model_file = os.path.join("/root", "real_model.json")
        experiment_path = os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id())
        if rank == 0:
            # trial get next parameter from network morphism tuner

            #开启模型生成和模型训练的异步执行
            lock = multiprocessing.Lock()
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.connect("tcp://172.17.0.9:800081")
            # trial get next parameter from network morphism tuner
            # path=os.environ["HOME"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/trials/" + str(nni.get_trial_id())
            os.makedirs(experiment_path + "/trials/" + str(nni.get_trial_id()))

            get_next_parameter_start = time.time()
            RCV_CONFIG = nni.get_next_parameter(socket)
            f11=open("/root/keras_trace"+str(rank),"a+")
            f11.write("rank-"+str(rank)+"-get_trial_id:"+str(nni.get_trial_id())+"\n")
            f11.close()
            get_next_parameter_end = time.time()

            while True:
                lock.acquire()
                f1 = open(experiment_path + "/graph.txt", "a+")
                f1.seek(0)
                lines = f1.readlines()
                f1.close()
                lock.release()
                if lines:
                    break
            json_and_id_str = lines[-1].replace("\n", "")  # 逆序读取并记录,数据组成字典
            json_and_id = dict((l.split('=') for l in json_and_id_str.split('+')))
            if str(json_and_id['history']) == "True":
                socket.send_pyobj({"type": "generated_parameter", "parameters": json_and_id['json_out'],
                                   "father_id": int(json_and_id['father_id']), "parameter_id": int(nni.get_sequence_id())})
                message = socket.recv_pyobj()
                f11=open("/root/send_to_dispatcher","a+")
                f11.write("json_and_id:"+str(json_and_id)+"\n")
                f11.close()
            elif str(json_and_id['history']) == "False":
                socket.send_pyobj({"type": "generated_parameter"})
                message = socket.recv_pyobj()

            RCV_CONFIG = json_and_id['json_out']
            f11=open("/root/log","a+")
            f11.write("########net generate \n")
            f11.write(str(RCV_CONFIG)+"\n")
            f11.close()
            parse_rev_args(RCV_CONFIG)
            f11=open("/root/log","a+")
            f11.write("########parse rev args finish \n")
            f11.close()
            with open(real_model_file, "w") as f:
                json.dump(RCV_CONFIG, f)
            # logger.info(RCV_CONFIG)
        else:
            while not os.path.isfile(real_model_file):
                time.sleep(5)
            with open(real_model_file, "r") as f:
                RCV_CONFIG = json.load(f)

        start_time = time.time()
        f = open(experiment_path + "/trials/" + str(nni.get_trial_id()) + "/output.log", "a+")
        f.write("sequence_id=" + str(nni.get_sequence_id()) + "\n")
        f.close()

        with open('search_space.json') as json_file:
            search_space = json.load(json_file)

        # 临时测试数据，后期改进可从参数中获取
        init_search_space_point = {"learning_rate": 0.1, "batch_size": 128}

        ## 根据father_id读取相应的超参json文件
        # 在起始时，不需要读取该件
        if 'father_id' in json_and_id:
            with open(experiment_path + '/hyperparameter/' + str(json_and_id['father_id']) + '.json') as hp_json:
                init_search_space_point = json.load(hp_json)
        # 初始化变量
        train_num = 0
        # 使用hyperopt_tuner中的API完成TPE超参搜索
        TPE = TPEtuner.HyperoptTuner('tpe')
        TPE.update_search_space(search_space)  # 输入space.json
        searched_space_point = {}
        # 执行第一次训练，获取训练时间，方便判断是否需要继续超参搜索
        train_time = time.time()
        start_date = time.strftime('%H.%M.%S', time.localtime(time.time()))

        ###############################################################################################################
        #执行第一次train和test
        ###############################################################################################################
        single_acc, current_ep = train_eval(init_search_space_point)

        single_train_time = time.time() - train_time
        best_final = single_acc
        f11=open("/root/keras_trace"+str(rank),"a+")
        f11.write("rank-"+str(rank)+"-single_acc:" + str(single_acc) + "\n")
        f11.close()
        searched_space_point = init_search_space_point

        # 记录当前trial的模型以及每次搜索的超参以及对应的epoch 2/3
        current_json = json_and_id['json_out']
        current_hyperparameter = init_search_space_point
        if not os.path.isdir(experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id())):
            os.makedirs(experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id()))
        with open(experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id()) + '/model.json', 'w') as f:
            f.write(current_json)
        with open(experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id()) + '/0.json', 'w') as f:
            json.dump({'hyperparameter': current_hyperparameter, 'epoch': current_ep, 'single_acc': single_acc,
                       'train_time': single_train_time, 'start_date': start_date}, f)

            # 从配置文件读取设备数
            #yml_f = open('/root/nni/examples/trials/network_morphism/cifar10/cifar10/config_singleGPU.yml')
            #yml = yaml.load(yml_f.read(), Loader=yaml.BaseLoader)
            if int(nni.get_sequence_id()) > 1:
                # if train_time > average_train_time:
                ##构造一个字典,先试试是否可行
                dict_first_data = init_search_space_point
                TPE.receive_trial_result(train_num, dict_first_data, single_acc)
                ## 增加TPE search 的早停机制
                TPEearlystop = utils.EarlyStopping(patience=5, mode="max")


        ###############################################################################################################
        #通过for循环执行后续的train和test
        #开始TPE超参搜索
        ###############################################################################################################
                for train_num in range(1, args.maxTPEsearchNum):
                    hy_train_start_time = time.time()
                    params = TPE.generate_parameters(train_num)
                    start_date = time.strftime('%H.%M.%S', time.localtime(time.time()))
                    single_acc, current_ep = train_eval(params)
                    TPE.receive_trial_result(train_num, params, single_acc)
                    hy_train_time = time.time() - hy_train_start_time
                    #                train_num = train_num+1

                    # 记录当前trial的模型以及每次搜索的超参以及对应的epoch 3/3
                    current_hyperparameter = params
                    with open(experiment_path + '/hyperparameter_epoch/' + str(nni.get_trial_id()) + '/' + str(
                            train_num) + '.json', 'w') as f:
                        json.dump(
                            {'hyperparameter': current_hyperparameter, 'epoch': current_ep, 'single_acc': single_acc,
                             'train_time': hy_train_time, 'start_date': start_date}, f)

                    if single_acc > best_final:
                        best_final = single_acc
                        searched_space_point = params
                    if TPEearlystop.step(single_acc):
                        break
            # 存储搜索出效果最好的超参
            if not os.path.isdir(experiment_path + '/hyperparameter'):
                os.makedirs(experiment_path + '/hyperparameter')
            with open(experiment_path + '/hyperparameter/' + str(nni.get_sequence_id()) + '.json',
                      'w') as hyperparameter_json:
                json.dump(searched_space_point, hyperparameter_json)

            end_time = time.time()

            f2 = open(experiment_path + "/c_time", "w+")
            f2.write(str(end_time - start_time))
            f2.close()
            f11=open('/root/keras_trace'+str(rank),'a+')
            f11.write("rank-"+str(rank)+"-end_time:" + str(end_time) + "\n")
            f11.write("rank-"+str(rank)+"-best_acc:" + str(best_final) + "\n")
            f11.close()
            f = open(experiment_path + "/trials/" + str(nni.get_trial_id()) + "/output.log", "a+")
            f.write("get_next_parameter_time=" + str(get_next_parameter_end - get_next_parameter_start) + "\n")
            f.write("example_time=" + str(time.time() - example_start_time) + "\n")
            f.write("duration=" + str(time.time() - start_time) + "\n")
            f.write("best_acc=" + str(best_final) + "\n")
            f.close()
            f11=open("/root/log","a+")
            f11.write("######result:"+str(best_final)+"\n")
            f11.close()
            # trial report best_acc to tuner
            if rank == 0:
                nni.report_final_result(best_final,socket)
                f11=open("/root/log","a+")
                f11.write("######need return master node\n")
                f11.close()
    except Exception as exception:
        f11=open("/root/log","a+")
        f11.write("######exception:"+str(exception)+"\n")
        f11.close()
        logger.exception(exception)
        raise
    exit(0)
