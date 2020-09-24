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
def save_log(result_path, experiments_id):
    log_path = os.path.join(result_path, 'logs')
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    os.system('cp -rf ' + os.environ["HOME"] + "/nni/experiments/" + experiments_id + '/checkpoint '+ log_path)
    os.system('cp -f ' + os.environ["HOME"] + "/mountdir/nni/experiments/" + experiments_id + '/graph.txt ' + log_path)
    os.system('cp -rf ' + os.environ["HOME"] + "/mountdir/nni/experiments/" + experiments_id + '/hyperparameter_epoch ' + log_path)
    os.system('cp -f ' + os.environ["HOME"] + "/nni/experiments/" + experiments_id + '/log/dispatcher.log '+ log_path)
    os.system('cp -f ' + os.environ["HOME"] + "/nni/experiments/" + experiments_id + '/log/nnimanager.log '+ log_path)
    os.system('cp -rf ' + os.environ["HOME"] + "/nni/experiments/" + experiments_id + '/trials '+ log_path)

def display_log(result):
    mat="{:^10}\t{:^10}\t{:^10}\t{:^10}"
    internal_log = "======================================================================\n"
    internal_log += str(mat.format("Time(H)", "Error(%)", "Score", "Regulated Score"))
    internal_log += "\n----------------------------------------------------------------------\n"
    for index in range(len(result['Error'])):
        internal_log +=  str(mat.format(result['real_time'][index], result['Error'][index], result['PFLOPS'][index], result['Score'][index]))
        internal_log += "\n"
    internal_log += "======================================================================\n"
    return internal_log
