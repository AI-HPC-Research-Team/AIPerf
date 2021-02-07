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

# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import argparse
import score
import numpy as np
import os
import save_log
import gen_report
from matplotlib.ticker import FuncFormatter

def formatnum(x, pos):
    return '$%.1f$x$10^{5}$' % (x/100000)
formatter = FuncFormatter(formatnum)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id",required = True, type=str, help="experiment_id")
    parser.add_argument("--logs", type=bool, default=False, help="whether to save log")
    return parser.parse_args()

def main_grid(time, values, save_folder, filename):
    time = np.array(time, dtype='float64')
    values = np.array(values, dtype='float64')
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(1,1,1)
    plt.plot(time, values, linewidth = '1.0',color='blue',marker='.') #'darkgoldenrod','slateblue','aqua','red','black'
    font = {'family':'DejaVu Sans', 'weight':'normal', 'size':12}
    plt.tick_params(labelsize=12)
    plt.xlabel('Hours',font)
    if filename == 'Score (in GFLOPS).png' or filename == 'Regulated Score (in GFLOPS).png':
        plt.gca().yaxis.set_major_formatter(formatter)
    plt.ylabel(filename.split('.')[0],font)
    plt.grid(axis="y")
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.yaxis.grid(True, which='major',linestyle='--')
    plt.savefig(os.path.join(save_folder,filename),format='png')
    plt.close()
    #plt.show()

def main(args, save_folder):
    results, trial_id_list, experiment_data = score.cal_report_results(args.id)
    main_grid(results['real_time'], results['GFLOPS'], save_folder, 'Score (in GFLOPS).png')
    main_grid(results['real_time'], results['Error'], save_folder, 'Error(%).png')
    main_grid(results['real_time'], results['Score'], save_folder, 'Regulated Score (in GFLOPS).png')
    errorth = 30.0
    logs = "======================================================================\n"
    if float(results['Error'][-1]) > errorth:
        logs += "!!! Test failed due to low accuracy !!!\n"

    logs += "Final Score : " + str(np.mean(np.array(results['GFLOPS'][-1],dtype='float64'))) + ' GFLOPS\n'
    logs += "Final Regulated Score : " + str(np.mean(np.array(results['Score'][-1],dtype='float64'))) + ' GFLOPS\n'

    internal_log = save_log.display_log(results)
    logs += internal_log
    print(logs)
    logs += str(results)
    logs += "\n----------------------------------------------------------------------\n"
    with open(os.path.join(save_folder, 'results.txt'), 'w') as txtfile:
        txtfile.write(logs)
    return results, trial_id_list, experiment_data

if __name__=='__main__':
    args = get_args()
    save_path = os.path.join(os.environ["HOME"], "mountdir/nni/experiments/")
    save_path = os.path.join(save_path, args.id)
    save_folder = os.path.join(save_path, 'results')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    results, trial_id_list, experiment_data = main(args, save_folder)
#     start_time = experiment_data[trial_id_list[0]][0][0][1]
#     for index in range(len(trial_id_list)-1,-1,-1):
#         if trial_id_list[index] in experiment_data:
#             stop_time = experiment_data[trial_id_list[index]][-1][-1][1]
#             break
#     gp = gen_report.GenPerfdata(start_time,stop_time)
#     gp.parse_mem()
#     gp.parse_cputil()
#     gp.parse_gpuutil()
#     gp.parse_gpumem()
    gen_report.GenReport(save_folder).output()
    print("Experiment report are saved in \"" + save_folder + '\"')
    os.system('rm ' + save_folder + '/*.png')
    if args.logs is True:
        save_log.save_log(save_folder, args.id)
    print("Finished.\n======================================================================\n")
