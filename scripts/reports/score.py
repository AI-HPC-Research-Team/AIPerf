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
import re
import json
import time
import datetime
import math
import profiler
import numpy as np

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
    epoch_size = 1

    # 遍历每一行
    for line in lines[start:end]+[lines[end]]:
        if 'PRINT' not in line:
            continue

        try:
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

            else:
                epoch_num = epoch_size

            if epoch_size < epoch_num:
                epoch_size = epoch_num
        except Exception as e:
            pass

    assert len(list(train_info.keys())) > 0, 'Train info is empty! Check param "train_num".'
    if len(list(eval_info.keys()))!=epoch_size or len(eval_info[epoch_size]['eval_time'])!=len(train_info[epoch_size]['epoch_time']):
        epoch_size -= 1

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


def find_all_trials(nnidir, expid, trial_id_list):
    experiment_data = {}

    for trial_id in trial_id_list:
        hp_list=[]
        filepath = os.path.join(nnidir, expid)
        filepath = os.path.join(filepath, 'trials')
        filepath = os.path.join(filepath, trial_id)
        filepath = os.path.join(filepath, 'trial.log')
        if not os.path.exists(filepath):
            continue

        # 获取每一个 train 的起始终止位置（一个 trial.log 可能包含两次 train 过程）
        lines = []
        info_end_list = [0]
        with open(filepath, 'r') as f:
            for index, line in enumerate(f.readlines()):
                lines.append(line)
                if 'HPO-' in line:
                    info_end_list.append(index)

        if len(lines)-info_end_list[-1] > 20:
            info_end_list.append(-1)
        elif info_end_list[-1]==0:
            continue
        else:
            pass

        # 依次解析每一个 train 的信息
        for start, end in zip(info_end_list[:-1], info_end_list[1:]):
            info = get_one_train_info(lines, start, end)
            eval_end_time = info['eval_end_time']
            eval_acc = info['eval_acc']

            res = []
            for index,(time_, acc_) in enumerate(zip(eval_end_time, eval_acc)):
                res.append([index+1, time_, acc_])

            hp_list.append(res)

        #print(trial_id, np.array(hp_list[-1]).shape)
        if np.array(hp_list[-1]).shape[0]==0:
            continue
        experiment_data[trial_id] = hp_list

    return experiment_data


def find_max_acc(stop_time, experiment_data):
    #print(trial)
    max_acc = 0
    lastest_time = 0
    result_dict = {}
    for trial in experiment_data:
        hy_count=0
        tmp_dict = {}
        for hp_data in experiment_data[trial]:
            hp_latest_time = 0
            hy_max_epoch = 0
            if hp_data:
                for epoch_data in hp_data:
                    if epoch_data[1] < stop_time:
                        #找该超参在停止时间内最高的正确率
                        if epoch_data[2] > max_acc:
                            max_acc = epoch_data[2]
                        #找该超参在停止时间内最晚完成的训练时间
                        if epoch_data[1] > lastest_time:
                            lastest_time = epoch_data[1]
                        #找该超参在停止时间内最大的epoch
                        if epoch_data[1] > hp_latest_time:
                            hp_latest_time = epoch_data[1]
                            if epoch_data[0] == 0:
                                continue
                            hy_max_epoch = epoch_data[0]
                if not hy_max_epoch == 0:
                    tmp_dict[hy_count] = hy_max_epoch
            hy_count+=1
        if tmp_dict:
            result_dict[trial] = tmp_dict
    return lastest_time, max_acc, result_dict


def conversion_time(string):
    '''
    转换时间,将每个trial下的trial.log记录的时间格式转换成时间戳，需要将AM和PM进行区分
    string字符串样式: "06/02/2020, 09:12:50 PM"
    '''
    if string.split()[1].split(':')[0] =='12':
        #trial.log记录的时间在12这个时间点与大众认知的时间不一样，直接换成时间戳会出问题，做一下处理
        if string.split()[2] == 'AM':
            date = string.split()[0]
            chj_time = ':'.join(['00',string.split()[1].split(':')[1], string.split()[1].split(':')[2]])
            time_system = string.split()[2]
            string = " ".join([date , chj_time , time_system])
        elif string.split()[2] == 'PM':
            string = string.split()[0] + ' ' + string.split()[1] + ' AM'

    #正常情况下string坐下切分可直接转换成时间戳
    mytime = string.split()[0][:-1]+' '+string.split()[1]
    mytime = time.mktime(time.strptime(mytime,"%m/%d/%Y %H:%M:%S"))

    #如果时间是PM还需要加上12小时, 再转换成时间戳
    if string.split()[2] == 'PM':
        datetime_struct = datetime.datetime.fromtimestamp(mytime)
        datetime_obj = (datetime_struct + datetime.timedelta( hours=12 ))
        mytime = datetime_obj.timestamp()     
    return mytime


def find_startime(trial_id_list, t, experiment_path):
    trial_id = trial_id_list[0]

    # 读取第一个trial 0号超参记录的开始时间
    if os.path.isfile(experiment_path + "/hyperparameter_epoch/" + trial_id + '/0.json'):
        with open(experiment_path + "/hyperparameter_epoch/" + trial_id + '/0.json') as hyperparameter_json:
            hyperparameter = json.load(hyperparameter_json)
    start_time = time.mktime(time.strptime(hyperparameter['start_date'], "%m/%d/%Y, %H:%M:%S"))

    # 将开始时间加上指定的结束时长，得到结束时间，转换成时间戳
    datetime_struct = datetime.datetime.fromtimestamp(start_time)
    datetime_obj = (datetime_struct + datetime.timedelta(hours=t))
    stop_time = datetime_obj.timestamp()

    return start_time,stop_time


def process_log(trial_id_list, experiment_data, dur, experiment_path):
    results = {}
    results['real_time'] = []
    results['PFLOPS'] = []
    results['Error'] = []
    results['Score'] = []

    flops_info = profiler.profiler(experiment_path)

    for index in range(1, int(dur)+2):
        start_time,stop_time = find_startime(trial_id_list, index, experiment_path)

        # 获取实验过程总数据
        lastest_time, max_acc, result_dict = find_max_acc(stop_time, experiment_data)
        # print(result_dict)

        # 开始计算
        run_sec = lastest_time - start_time
        # print(datetime.datetime.fromtimestamp(start_time),'\t',datetime.datetime.fromtimestamp(stop_time),'\t',stop_time-start_time)

        total_FLOPs = 0
        faild_trial = []
        for i in range(len(flops_info['trialid'])):
            trial_id = flops_info['trialid'][i]
            if trial_id in result_dict:
                hp_num = int(flops_info['hpoid'][i])
                eval_ops = float(flops_info['eval_per_image'][i])
                trian_ops = float(flops_info['train_per_image'][i])
                #判断在截止时间前，是否已经产生该超参
                if hp_num in result_dict[trial_id].keys():
                    #读取每个超参对应的epoch
                    epoch = result_dict[trial_id][hp_num]
                    total_FLOPs += (eval_ops * 50000 + trian_ops * 1280000) * epoch
                    #print(trial_id,hp_num,epoch)
            else:
                faild_trial.append(trial_id)
        fraction = float(float(total_FLOPs) * float(abs(math.log(1-max_acc,math.e)))) / float(run_sec)
        fraction = fraction / (10**15)

        results['real_time'].append('{:.2f}'.format(run_sec / 3600.))
        results['PFLOPS'].append('{:.4f}'.format(float(total_FLOPs) / float(run_sec) / (10**15)))
        results['Error'].append('{:.2f}'.format(100 - max_acc * 100))
        results['Score'].append('{:.4f}'.format(fraction))
    return results

def cal_report_results(expid):
    id_dict = {}
    nnidir = os.path.join(os.environ["HOME"], "nni/experiments/")
    mountdir = os.path.join(os.environ["HOME"], "mountdir/nni/experiments/")
    experiment_path = os.path.join(mountdir, expid)

    #获取sequence_id和trial_id，根据sequence_id从大到小排序
    for trials in os.listdir(os.path.join(nnidir, expid,'trials')):
        with open(os.path.join(nnidir, expid,'trials',trials,'parameter.cfg'), "r") as file_read:
            json_read = json.load(file_read)
            parameter_id = json_read['parameter_id']
            id_dict[parameter_id] = trials

    #根据 sequence_id 由大到小排序 id_dict = {sequence_id : trial_id}
    id_dict = sorted(zip(id_dict.keys(),id_dict.values()))
    id_dict = dict(id_dict)
    trial_id_list = list(id_dict.values())


    experiment_data = find_all_trials(nnidir, expid, trial_id_list)
    start_time = experiment_data[trial_id_list[0]][0][0][1]
    for index in range(len(trial_id_list)-1,-1,-1):
        if trial_id_list[index] in experiment_data:
            stop_time = experiment_data[trial_id_list[index]][-1][-1][1]
            break
    dur = (stop_time - start_time) / 3600.
    results = process_log(trial_id_list, experiment_data, dur, experiment_path)
    return results, trial_id_list, experiment_data
