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
import time
import multiprocessing
import psutil
import os

#run_time = 24
net_interval = 30
dev_interval = 30

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id',type=str,required = True,help="experiment id")
    return parser.parse_args()

def test_record_device_info(experiment_id):
    count = 0
    while True:
        count+=1
        with open('/root/' + experiment_id,'a+') as f:
            f.write(str(count)+'\n')
        time.sleep(2)

def record_net_info():
    u_before = psutil.net_io_counters().bytes_sent
    d_before = psutil.net_io_counters().bytes_recv
    def get_send(u_before):
        u_now = psutil.net_io_counters().bytes_sent
        upload = (u_now - u_before)
        u_before = u_now
        return upload,u_before
    def get_recv(d_before):
        d_now = psutil.net_io_counters().bytes_recv
        download = (d_now - d_before)
        d_before = d_now
        return download,d_before
    #while((time.time() - start_time)/3600 < run_time):
    while True:
        upload,u_before =  get_send(u_before)
        download,d_before = get_recv(d_before)
        f=open(log_path + '/net_info.csv','a+')
        f.write(str(upload) + ',' + str(download) + '\n')
        f.close()
        time.sleep(net_interval)

def write_file(path,content):
    record_file = open(path,'a+')
    record_file.write(content + '\n')
    record_file.close()


def record_device_info():
    #while((time.time() - start_time)/3600 < run_time):
    while True:
        cpu_core_context =str(psutil.cpu_percent(interval=1, percpu=True))
        write_file(log_path + '/cpu_core_info.csv',cpu_core_context)
        cpu = os.popen("export TERM=linux && top -bn 2|grep Cpu\(s\)|awk '{print $2+$4}'").readlines()[1].strip()
        write_file(log_path + '/cpu_info.csv',cpu)
        mem = os.popen("export TERM=linux && free |grep -E 'Mem:' |awk '{print $3}' ").readline().strip()
        write_file(log_path + '/mem_info.csv',str(int(mem)/1024)) #M
        #gpu = os.popen("nvidia-smi  dmon -c 1 -s u|awk '{if($1 ~ /^[0-9]+$/){print $1,$2,$3}}'").readlines()
        #for i in gpu:
        #    file_name = str(i.strip().split(' ')[0]) + '.csv'
        #    content = str(','.join((i.strip().split(' ')[1],i.strip().split(' ')[2])))
        #    write_file(log_path + '/' + file_name,content)
        time.sleep(dev_interval - 2)

if __name__ == "__main__":
    args = get_args()
    experiment_id = args.id
    log_path = '/root/mountdir/device_info/' + experiment_id + '/' + os.environ['SLURMD_NODENAME']
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    start_date = os.popen('date').readline().strip()
    write_file(log_path + '/time',start_date)

    start_time = time.time()

    dev_p = multiprocessing.Process(target=record_device_info)
    dev_p.start()
    net_p = multiprocessing.Process(target = record_net_info)
    net_p.start()
    dev_p.join()
    net_p.join()
