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

import numpy as np

import os
import psutil
import hashlib
import time, datetime
#from tensorflow.python.client import device_lib as _device_lib
from numpy import log2
from scipy.optimize import curve_fit

class EarlyStopping:
    """ EarlyStopping class to keep NN from overfitting
    """

    # pylint: disable=E0202
    def __init__(self, mode="min", min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        """ EarlyStopping step on each epoch
        Arguments:
            metrics {float} -- metric value
        """

        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


def trial_activity(file_path, ppid):
    '''判断当前trial是否为卡死状态，如果为卡死状态则直接结束该trial；判断调条件为:
       每隔2.5分钟获取一次:
           1.trial对应的trial.log的哈希值判断是否相等;
           2.获取trial进程的cpu占用，如果连续10秒都为0，则判断为0;
           3.获取trial进程的所有gpu占用，如果连续5秒所有gpu都为0，则判断为0;
       以上条件连续三次都成立则判断trial为卡死状态。
    '''

    def get_md5(file_path):
        '''获取日志文件的哈希值'''
        m = hashlib.md5()
        with open(file_path, 'rb') as f:
            for line in f:
                m.update(line)
        md5code = m.hexdigest()
        return md5code

    def ppid_cpu_info(ppid):
        '''判断主进程cpu占用是否为0,如果不为0则返回1，连续采样10秒都为0，则返回0'''
        p_info = psutil.Process(ppid)
        for i in range(10):
            if p_info.cpu_percent() != 0:
                return 1
            time.sleep(1)
        return 0

    def ppid_gpu_info():
        '''判断主进程所占用的gpu使用率是否为0,如果不为0则返回1，连续采样5秒都为0，则返回0'''
        local_device_protos = _device_lib.list_local_devices()
        # 获取当前进程可用的gpu索引
        index = ','.join([x.name.split(':')[2] for x in local_device_protos if x.device_type == 'GPU'])
        for i in range(5):
            gpu_info = os.popen(
                "nvidia-smi  dmon -c 1 -s u -i " + index + "|awk '{if($1 ~ /^[0-9]+$/){print $2}}'").readlines()
            for i in gpu_info:
                if int(i.strip()) != 0:
                    return 1
            time.sleep(1)
        return 0

    sample_interval = 2.5 * 60
    while True:
        before_md5 = get_md5(file_path)
        before_cpu = ppid_cpu_info(ppid)
        before_gpu = ppid_gpu_info()
        time.sleep(sample_interval)
        if before_cpu != 0 or before_gpu != 0:
            continue
        after_md5 = get_md5(file_path)
        after_cpu = ppid_cpu_info(ppid)
        after_gpu = ppid_gpu_info()
        time.sleep(sample_interval)
        if after_cpu != 0 or after_gpu != 0:
            continue
        elif before_md5 == after_md5:
            latest_md5 = get_md5(file_path)
            latest_cpu = ppid_cpu_info(ppid)
            latest_gpu = ppid_gpu_info()
            if after_md5 == latest_md5 and after_cpu == latest_cpu and after_gpu == latest_gpu:
                break
            else:
                time.sleep(sample_interval)
    os.system('kill -9 ' + str(ppid))


def predict_acc(tid, epoch_x, acc_y, epoch=75, saveflag=False, totaly=[]):
    """

    Parameters
    ----------
    epoch_x: epoch列表
    acc_y：epoch对应的acc列表
    epoch：待预测acc的epoch数
    drawflag：是否将预测图保存
    totaly

    Returns 预测epoch数的acc结果
    -------

    """

    def logfun(x, a, b, c):
        x = np.array(x)
        y = a * log2(x + b) + c
        return y
    
    epoch_x = list(epoch_x)
    acc_y = list(acc_y)
    
    epoch_x.append(120)
    acc_y.append(0.72)
    
    popt, pcov = curve_fit(logfun, epoch_x, acc_y, maxfev=10000)

    acc_y_true_numformat = np.array(acc_y, dtype=float)
    acc_y_predict = logfun(epoch_x, popt[0], popt[1], popt[2])

    error = np.sqrt(np.sum(np.square(np.subtract(acc_y_true_numformat, acc_y_predict))) / len(acc_y_predict))
    # print(error)
    acc_epoch_target = logfun(epoch, popt[0], popt[1], popt[2])

    acc_epoch_target_minus_error = acc_epoch_target - 2 * error
    if acc_epoch_target_minus_error < max(acc_y) or acc_epoch_target_minus_error > 1:
        acc_epoch_target = max(acc_y)
        labletxt = "x-%d,no-pred,wpred-%5.3f,max-%5.3f,err-%5.3f,l-%s" % (
            epoch, acc_epoch_target_minus_error, acc_epoch_target, error, str(len(acc_y)))
    else:
        acc_epoch_target = acc_epoch_target_minus_error
        labletxt = "x-%d,use-pred, tpred%5.3f,err-%5.3f,l-%s" % (epoch, acc_epoch_target, error, str(len(acc_y)))

    print("Predicted_val_acc:"+str(acc_epoch_target))
    return acc_epoch_target


def conversion_time(string):
    '''
    转换时间,将每个trial下的trial.log记录的时间格式转换成时间戳，需要将AM和PM进行区分
    string字符串样式: "06/02/2020, 09:12:50 PM"
    '''
    if string.split()[1].split(':')[0] == '12':
        # trial.log记录的时间在12这个时间点与大众认知的时间不一样，直接换成时间戳会出问题，做一下处理
        if string.split()[2] == 'AM':
            date = string.split()[0]
            chj_time = ':'.join(['00', string.split()[1].split(':')[1], string.split()[1].split(':')[2]])
            time_system = string.split()[2]
            string = " ".join([date, chj_time, time_system])
        elif string.split()[2] == 'PM':
            string = string.split()[0] + ' ' + string.split()[1] + ' AM'

    # 正常情况下string坐下切分可直接转换成时间戳
    mytime = string.split()[0][:-1] + ' ' + string.split()[1]
    mytime = time.mktime(time.strptime(mytime, "%m/%d/%Y %H:%M:%S"))

    # 如果时间是PM还需要加上12小时, 再转换成时间戳
    if string.split()[2] == 'PM':
        datetime_struct = datetime.datetime.fromtimestamp(mytime)
        datetime_obj = (datetime_struct + datetime.timedelta(hours=12))
        mytime = datetime_obj.timestamp()
    return mytime


# 从指定位置的 trial.log 文件中解析一个 train 的信息
def get_one_train_info(lines, start, end):
    '''
    :param lines:
    :param start:
    :param end:
    :return:
    '''
    train_info = dict()
    eval_info = dict()

    # 遍历每一行
    for line in lines[start:end]:
        if 'PRINT' not in line:
            continue
        # 获取每一行的打印时间
        print_time, print_info = line.split('PRINT')
        print_time = print_time.strip().strip('[').strip(']')
        print_info = print_info.strip()
        print_time = conversion_time(print_time)

        # 如果是训练信息，则提取：epoch训练时间，step训练时间，loss
        if 'train time' in print_info:
            epoch_num, epoch_time, step_time, loss = print_info.split(', ')

            epoch_num = int(epoch_num.split('Epoch')[-1].strip().split('/')[0])
            epoch_time = float(epoch_time.split('train time:')[-1].strip()) / 1000
            step_time = float(step_time.split('per step time:')[-1].strip()) / 1000
            loss = float(loss.split('loss:')[-1].strip())

            if epoch_num in list(train_info.keys()):
                train_info[epoch_num]['epoch_time'].append(epoch_time)
                train_info[epoch_num]['step_time'].append(step_time)
                train_info[epoch_num]['loss'].append(loss)
            else:
                train_info[epoch_num] = dict()
                train_info[epoch_num]['epoch_time'] = [epoch_time]
                train_info[epoch_num]['step_time'] = [step_time]
                train_info[epoch_num]['loss'] = [loss]

        # 如果是验证信息，则提取：验证精度，验证时间，验证结束时间
        elif 'EvalTime' in print_info:
            epoch_num, eval_acc, eval_time = print_info.split(', ')

            epoch_num = int(epoch_num.split('Epoch')[-1].strip().split('/')[0])
            eval_acc = float(eval_acc.split('EvalAcc:')[-1].strip())
            eval_time = float(eval_time.strip('s').split('EvalTime')[-1].strip())

            if epoch_num in list(eval_info.keys()):
                eval_info[epoch_num]['eval_acc'].append(eval_acc)
                eval_info[epoch_num]['eval_time'].append(eval_time)
                eval_info[epoch_num]['eval_end_time'].append(print_time)
            else:
                eval_info[epoch_num] = dict()
                eval_info[epoch_num]['eval_acc'] = [eval_acc]
                eval_info[epoch_num]['eval_time'] = [eval_time]
                eval_info[epoch_num]['eval_end_time'] = [print_time]
    assert len(list(train_info.keys())) > 0, 'Train info is empty! Check param "train_num".'
    epoch_size = max(list(train_info.keys()))

    epoch_time_list = []
    step_time_list = []
    loss_list = []
    eval_acc_list = []
    eval_time_list = []
    eval_end_time_list = []

    # 汇总所有线程信息，得到每一个epoch的训练、验证信息
    for i in range(epoch_size):
        epoch_time_list.append(max(train_info[i + 1]['epoch_time']))
        step_time_list.append(max(train_info[i + 1]['step_time']))
        loss_list.append(np.mean(train_info[i + 1]['loss']))
        eval_acc_list.append(np.mean(eval_info[i + 1]['eval_acc']))
        eval_time_list.append(max(eval_info[i + 1]['eval_time']))
        eval_end_time_list.append(max(eval_info[i + 1]['eval_end_time']))

    return {'epoch_train_time': epoch_time_list,
            'step_train_time': step_time_list,
            'loss': loss_list,
            'eval_acc': eval_acc_list,
            'eval_time': eval_time_list,
            'eval_end_time': eval_end_time_list}

def MinGpuMem():
    gpu_mem_list = os.popen("nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits").readlines()
    gpu_mem_list = list(map(lambda x:int(x.strip()), gpu_mem_list))
    min_gpu_mem = round(min(gpu_mem_list)/1024.0)
    return int(min_gpu_mem)
