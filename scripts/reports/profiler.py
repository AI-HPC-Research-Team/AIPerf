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

import json
import argparse
import os

coeff = {"add": 1, "mul": 1, "macc": 2, "div": 4, "sqrt": 4, "exp": 8, "comp": 1, "log": 1, "sin": 1}
image_num = 1

def cal_conv_flops(in_channel, out_channel, kernel_size, out_size, flag):
    macc = (out_size * out_size * kernel_size * kernel_size * in_channel) * out_channel
    fwflops = macc * coeff['macc']
    params = (in_channel * kernel_size * kernel_size + 1) * out_channel
    if flag == 0:
        bwflops = kernel_size * kernel_size * in_channel * out_channel * (2 * out_size * out_size + 2)
    else:
        bwflops = kernel_size * kernel_size * in_channel * out_channel * (4 * out_size * out_size + 2)

    fwflops *= image_num
    bwflops *= image_num
    return fwflops, macc, params, bwflops

def cal_dense_flops(in_channel, out_channel):
    macc = in_channel * out_channel
    fwflops = macc * coeff['macc']
    params = (in_channel + 1) * out_channel
    bwflops = 2 * (2 * in_channel * out_channel) + 2 * (in_channel + 1) * out_channel

    fwflops *= image_num
    bwflops *= image_num
    return fwflops, macc, params, bwflops

def cal_bn_flops(in_channel, in_size):
    macc = in_size[0] * in_size[1] * in_channel
    div = in_size[0] * in_size[1] * in_channel
    add = in_size[0] * in_size[1] * in_channel
    fwflops = macc * coeff['macc'] + add * coeff['add'] + div * coeff['div']
    params = in_channel * 2
    bwflops = (29.0 * image_num + 7) / image_num

    fwflops *= image_num
    bwflops *= image_num
    return fwflops, macc, params, bwflops

def cal_relu_flops(input_size):
    fwflops = (input_size[0] * input_size[1] * input_size[2]) * coeff['comp']
    params = 0

    fwflops *= image_num
    bwflops = 0
    return fwflops, params, bwflops

def cal_add_flops(input_size):
    fwflops = (input_size[0] * input_size[1] * input_size[2]) * coeff['add']
    params = 0

    fwflops *= image_num
    bwflops = 0
    return fwflops, params, bwflops

def cal_maxpool_flops(input_size, kernel_size, stride):
    fwflops = (1.0 * input_size[0] * input_size[1] * input_size[2] * kernel_size * kernel_size / stride / stride) * coeff['comp']
    params = 0

    fwflops *= image_num
    bwflops = 0
    return fwflops, params, bwflops

def cal_avgpool_flops(input_size):
    fwflops = (input_size[0] * input_size[1] - 1) * input_size[2] * coeff['add'] + input_size[2] * coeff['div']
    params = 0

    fwflops *= image_num
    bwflops = 0
    return fwflops, params, bwflops

def cal_softmax_flops(out_channel):
    fwflops = out_channel * coeff['exp'] + (out_channel - 1) * coeff['add'] + out_channel * coeff['div']
    params = 0

    fwflops *= image_num
    bwflops = 0
    return fwflops, params, bwflops

def cal_trial_flops_per_image(model_name, hyper_name):
    model = open(model_name, 'r')
    model_str = json.load(model)
    model.close()
    hyper = open(hyper_name, 'r')
    hyper_str = json.load(hyper)
    hyper.close()

    hpo_kernel_size = hyper_str['hyperparameter']['kernel_size']
    dropout_rate = hyper_str['hyperparameter']['dropout_rate']
    flag = 0
    count = 0
    input_shape = model_str['node_list']
    layers = model_str['layer_list']
    fwconvflops = 0
    fwdenseflops = 0
    fwbnflops = 0
    fwreluflops = 0
    fwaddflops = 0
    fwconcatflops = 0
    fwmaxpoolflops = 0
    fwsoftmaxflops = 0
    fwavgpoolflops = 0

    bwconvflops = 0
    bwdenseflops = 0
    bwbnflops = 0
    bwreluflops = 0
    bwaddflops = 0
    bwconcatflops = 0
    bwmaxpoolflops = 0
    bwsoftmaxflops = 0
    bwavgpoolflops = 0

    macc = 0
    params = 0
    for index in range(len(layers)):
        layer = layers[index][1]
        count += 1
        if layer[0] == 'StubConv2d':
            in_node = layer[1]
            input_size = input_shape[in_node][1]
            in_size = input_size
            in_channel = layer[3]
            out_channel = layer[4]
            kernel_size = layer[5]
            if (count > 175):
                kernel_size = int(hpo_kernel_size)
            stride = layer[6]
            padding = layer[7]
            out_size = (input_size[0] + 2 * padding - kernel_size)//stride + 1
            fwtempflops, tempmacc, tempparams, bwtempflops = cal_conv_flops(in_channel, out_channel, kernel_size, out_size, flag)
            fwconvflops += fwtempflops
            macc += tempmacc
            params += tempparams
            bwconvflops += bwtempflops
            flag += 1

        if layer[0] == 'StubDense':
            in_node = layer[1]
            input_size = input_shape[in_node][1]
            in_size = input_size
            in_channel = layer[3]
            out_channel = layer[4]
            in_channel = int(in_channel)
            fwtempflops, tempmacc, tempparams, bwtempflops = cal_dense_flops(in_channel, out_channel)
            fwdenseflops += fwtempflops
            bwdenseflops += bwtempflops
            params += tempparams

            fwtempflops, tempparams, bwtempflops = cal_softmax_flops(out_channel)  # softmax has no BP flops
            fwsoftmaxflops += fwtempflops
            bwsoftmaxflops += bwtempflops
            params += tempparams

        if layer[0] == 'StubBatchNormalization2d':
            in_node = layer[1]
            input_size = input_shape[in_node][1]
            in_size = input_size
            in_channel = layer[3]
            fwtempflops, tempmacc, tempparams, bwtempflops = cal_bn_flops(in_channel, in_size)
            fwbnflops += fwtempflops
            macc += tempmacc
            params += tempparams
            bwbnflops += bwtempflops

        if layer[0] == 'StubReLU':
            in_node = layer[1]
            input_size = input_shape[in_node][1]
            in_size = input_size
            fwtempflops, tempparams, bwtempflops = cal_relu_flops(in_size)
            fwreluflops += fwtempflops
            params += tempparams
            bwreluflops += bwtempflops
            continue

        if layer[0] == 'StubPooling2d':
            in_node = layer[1]
            input_size = input_shape[in_node][1]
            in_size = input_size
            kernel_size = layer[3]
            stride = layer[4]
            fwtempflops, tempparams, bwtempflops = cal_maxpool_flops(in_size, kernel_size, stride)
            fwmaxpoolflops += fwtempflops
            params += tempparams
            bwmaxpoolflops += bwtempflops
            continue

        if layer[0] == 'StubGlobalPooling2d':
            in_node = layer[1]
            input_size = input_shape[in_node][1]
            in_size = input_size
            fwtempflops, tempparams, bwtempflops = cal_avgpool_flops(in_size)
            fwavgpoolflops += fwtempflops
            params += tempparams
            bwavgpoolflops += bwtempflops
            continue

        if layer[0] == 'StubAdd':
            in_node = layer[1]
            input_size1 = input_shape[in_node[0]][1]
            input_size2 = input_shape[in_node[1]][1]
            if input_size1 == input_size2:
                fwtempflops, tempparams, bwtempflops = cal_add_flops(input_size1)
                fwaddflops += fwtempflops
                params += tempparams
                bwaddflops += bwtempflops
            else:
                # does not contribute to flops, in this node, two input nodes are already in the same shape
                continue

        if layer[0] == 'StubConcatenate':
            # does not contribute to flops, in this node, two input nodes are already in the same shape
            continue

    fwflops = fwconvflops + fwdenseflops + fwbnflops + fwreluflops + fwaddflops + fwconcatflops + fwsoftmaxflops + fwmaxpoolflops + fwavgpoolflops
    bwflops = bwconvflops + bwdenseflops + bwbnflops + bwreluflops + bwaddflops + bwconcatflops + bwsoftmaxflops + bwmaxpoolflops + bwavgpoolflops

    eval_flops_per_image = fwflops
    train_flops_per_image = bwflops + fwflops

    return eval_flops_per_image, train_flops_per_image

def profiler(exppath):
    trialdir = os.path.join(exppath,'hyperparameter_epoch')

    tempdict = {}
    tempdict['trialid'] = []
    tempdict['hpoid'] = []
    tempdict['eval_per_image'] = []
    tempdict['train_per_image'] = []
    for trial in os.listdir(trialdir):
        trialpath = os.path.join(trialdir, trial)
        files = os.listdir(trialpath)
        if 'model.json' in files:
            model_name = os.path.join(trialpath, 'model.json')
            for hpofile in files:
                if not hpofile == "model.json":
                    hpoid = hpofile.split('.')[0]
                    hyper_name = os.path.join(trialpath, hpofile)
                    eval_flops_per_image, train_flops_per_image = cal_trial_flops_per_image(model_name, hyper_name)
                    tempdict['trialid'].append(trial)
                    tempdict['hpoid'].append(hpoid)
                    tempdict['eval_per_image'].append(eval_flops_per_image)
                    tempdict['train_per_image'].append(train_flops_per_image)

    #print("total models: "+str(len(tempdict['trialid'])))
    return tempdict

