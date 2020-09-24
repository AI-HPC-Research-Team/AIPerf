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

# coding:utf-8

import sys
import json
import time

class ModifyJson():
    def __init__(self,ir_model_json, esargs):
        self.ir_model_json = ir_model_json
        self.esargs = esargs

    def modify_json_file(self,json_dict, index, layer_str, param):
        layer_list = json_dict["layer_list"]
        item = layer_list[index]

        if item[1][0] == "StubDropout2d":
            item[1][3] = param 

        if item[1][0] == "StubConv2d":
            item[1][5] = param

        return json_dict

    def modify_hyper_parameters(self):
        json_dict = json.loads(self.ir_model_json)
        layer_list = json_dict["layer_list"] 
        denseList = []

        for index, layer in layer_list:
            if layer[0] == "StubDropout2d":
                layer_str = "StubDropout2d"
                json_dict = self.modify_json_file(json_dict, index, layer_str, self.esargs['dropout_rate'])

            if layer[0] == "StubDense":
                denseList.append(index)

        if len(denseList) == 0:
            thresh = 175
        elif len(denseList) > 0:
            thresh = denseList[-1]

        for index1, layer1 in layer_list:
            if index1 > thresh:
                if layer1[0] == "StubConv2d":
                    layer_str = "StubConv2d"
                    json_dict = self.modify_json_file(json_dict, index1, layer_str, self.esargs['kernel_size'])

        temp = json.dumps(json_dict)

        return temp

