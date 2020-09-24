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
import torch
import torch.distributed as dist
from torch.nn import Module
import multiprocessing as mp

class DistModule(Module):
    def __init__(self, module, broadcast=True):
        super(DistModule, self).__init__()
        self.module = module
        if broadcast:
            broadcast_params(self.module)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def __getattr__(self, attr):
        try:
            return super(DistModule, self).__getattr__(attr)
        except:
            return getattr(self.module, attr)

def average_gradients(model):
    """ sum gradients """
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data)

def broadcast_params(model):
    """ broadcast model parameters """
    for p in model.state_dict().values():
        dist.broadcast(p, 0)

def dist_init(port):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    proc_id = int(os.environ['SLURM_PROCID'])
    local_id = int(os.environ['SLURM_LOCALID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    f11=open('/root/log','a+')
    f11.write('######os.environ+ get gpu num:'+str(num_gpus)+"\n")
    f11.write('######os.environ+ ntask num:'+str(ntasks)+"\n")
    f11.close()
    #print('local_id', local_id)
    #print(node_list, ntasks)
    torch.cuda.set_device(local_id%num_gpus)

    # if '[' in node_list:
    #     beg = node_list.find('[')
    #     pos1 = node_list.find('-', beg)
    #     if pos1 < 0:
    #         pos1 = 1000
    #     pos2 = node_list.find(',', beg)
    #     if pos2 < 0:
    #         pos2 = 1000
    #     node_list = node_list[:min(pos1,pos2)].replace('[', '')
    # # addr = node_list[8:].replace('-', '.')
    # addr = node_list
    # print(addr)
    addr = os.environ['SLURMD_NODENAME']

    os.environ['MASTER_PORT'] = port
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size
