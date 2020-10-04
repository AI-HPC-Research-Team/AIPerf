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

# encoding=utf-8
"""
#说明
# convert to pdf method 1, not support ZN and html css style
# step1: pip install xhtml2pdf
# example:
# from xhtml2pdf import pisa
# with open('output/report.pdf', "w+b") as out_pdf_file_handle:
#     pisa.CreatePDF(
#         src=html,  # HTML to convert
#         dest=out_pdf_file_handle)

# convert to pdf method 2, not support img link source, need use base64 code
# step1:pip install pdfkit
# step2:install wkhtmltopdf, website https://wkhtmltopdf.org/downloads.html
# example:
# import pdfkit
# pdfkit.from_file('output/report_time.html','output/report.pdf')
"""
import os
import base64
import datetime
from jinja2 import Environment
from jinja2 import FileSystemLoader
import requests
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import csv

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
PATH = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = PROJECT_ROOT  # html模板路径

TEMPLATE_ENVIRONMENT = Environment(
    autoescape=False,
    loader=FileSystemLoader(TEMPLATE_PATH),
    trim_blocks=False)


class GenReport(object):
    """
    格式化输出报告
    """

    def __init__(self,path):
        self.resultdict = {}
        self.t_str, self.t_str_show = self.timenow()
        self.status = "Succeed"
        self.path = path

    def getbase64(self, file):
        """
        获取png的base64码便于转为pdf
        """
        icon = open(file, 'rb')
        iconData = icon.read()
        iconData = base64.b64encode(iconData)
        return ("data:image/jpg;base64," + str(iconData)).replace("b'", '').replace("'", "")

    def fillresult(self, resultdict):
        """
        填充html需要的数据
        """
        resultdict["nowtime"] = self.t_str_show
        txtcontent = open(os.path.join(self.path, "results.txt")).read()
        nfindex = txtcontent.index("\n")
        firstline = txtcontent[0:nfindex]
        showcontent = txtcontent.split(firstline + "\n")[1]
        resultdict["error"] = self.getbase64(os.path.join(self.path, "Error(%).png"))
        resultdict["flops"] = self.getbase64(os.path.join(self.path, "Score (in GFLOPS).png"))
        resultdict["score"] = self.getbase64(os.path.join(self.path, "Regulated Score (in GFLOPS).png"))
        resultdict["warning"] = 0
        if "failed" in showcontent:
            self.status = "Failed"
            resultdict["warning"] = 1
            resultdict["warning_info"] = showcontent.split("\n")[0]
            conclusion = showcontent.split("\n")[1:]
            clist = []
            for c in conclusion:
                if c == "":
                    continue
                clist.append([c.split(":")[0],c.split(":")[1]])
            resultdict["conclusion"] = clist
        else:
            conclusion = showcontent.split("\n")
            clist = []
            for c in conclusion:
                if c == "":
                    continue
                clist.append([c.split(":")[0],c.split(":")[1]])

            resultdict["conclusion"] = clist

    def render_template(self, template_filename, context):
        """
        进行html模板填充
        """
        return TEMPLATE_ENVIRONMENT.get_template(template_filename).render(context)

    def gen_pdf(self):
        """
        html转pdf
        """
        import pdfkit
        pdfkit.from_file(os.path.join(self.path, 'Report_%s_%s.html' % (self.t_str, self.status)),
                         os.path.join(self.path, 'Report_%s_%s.pdf' % (self.t_str, self.status)))

    def timenow(self):
        now = datetime.datetime.now()
        return now.strftime('%Y%m%d%H%M%S'), now.strftime('%Y-%m-%d %H:%M:%S')  # %b-%m-%y %H:%M:%S

    def output(self, pdf=False):
        self.fillresult(self.resultdict)
        html_path = os.path.join(self.path, "Report_%s_%s.html" % (self.t_str, self.status))
        with open(html_path, 'w', encoding="utf-8") as f:
            html = self.render_template('template_result.html', self.resultdict)
            f.write(html)

        if pdf:
            self.gen_pdf()


# 1-avg(irate(node_cpu_seconds_total{instance="localhost:9100",mode="idle"}[5m]))
class GenPerfdata(object):
    URL = "http://10.133.75.11:9090/api/v1/query_range"
    CPU_UTIL = "1 - avg(irate(node_cpu_seconds_total{instance='%s:9100',mode='idle'}[5m]))"
    MEM = "(node_memory_MemTotal_bytes{instance='%s:9100'} - node_memory_MemAvailable_bytes{instance='%s:9100'})/node_memory_MemTotal_bytes{instance='%s:9100'}"
    GPU_UTIL = "DCGM_FI_DEV_GPU_UTIL"
    GPU_MEM = "DCGM_FI_DEV_FB_USED/(DCGM_FI_DEV_FB_FREE+DCGM_FI_DEV_FB_USED)"

    def __init__(self, start=1596464987.015, end=1596189014.015, step=30):
        self.params = {"start": start, "end": self.timenow(), "step": step}
        self.slave = []

    def timeint(self, t):
        pass

    def timenow(self):
        import datetime
        import time
        return time.mktime(datetime.datetime.now().timetuple())

    def getmix_start(self, data):
        tosortlist = []
        for k, v in data.items():
            if len(v[0]) == 2:
                tosortlist.append({"slave": k, "type": "0", "time": v[0][0]})
                tosortlist.append({"slave": k, "type": "1", "time": v[-1][0]})
            else:  # 多个gpu场景
                tosortlist.append({"slave": k, "type": "0", "time": v[0][0][0]})
                tosortlist.append({"slave": k, "type": "1", "time": v[0][-1][0]})

        tosortlist.sort(key=lambda x: (x["time"], x["type"]))
        llen = int(len(tosortlist) / 2)
        start = tosortlist[llen - 1]["time"]
        end = tosortlist[llen]["time"]
        print(start, end)
        return start, end

    def align(self, data):
        """
        对齐数据
        """
        start, end = self.getmix_start(data)
        datalen = 0
        avg_denominator = 0
        for slave in self.slave:
            tvalues = data[slave]
            if len(data[slave][0]) == 2:
                for i, tv in enumerate(tvalues):
                    if start == tv[0]:
                        istart = i
                    if end == tv[0]:
                        if len(tvalues) - 1 > i:
                            data[slave] = data[slave][istart:(i + 1)]
                            datalen = len(data[slave])
                            avg_denominator += 1
                        else:
                            data[slave] = data[slave][istart:]
                            datalen = len(data[slave])
                            avg_denominator += 1
            else:
                for i, tv in enumerate(tvalues[0]):
                    if start == tv[0]:
                        istart = i
                    if end == tv[0]:
                        if len(tvalues) - 1 > i:
                            mlist = len(data[slave])
                            for m in range(0, mlist):
                                data[slave][m] = data[slave][m][istart:(i + 1)]
                                avg_denominator += 1
                            datalen = len(data[slave][0])
                            break
                        else:
                            mlist = len(data[slave])
                            for m in range(0, mlist):
                                data[slave][m] = data[slave][m][istart:]
                                avg_denominator += 1
                            datalen = len(data[slave][0])
                            break
        return data, datalen, avg_denominator

    def cal_avg(self, data, per=False, span=1):
        """
        计算均值
        """
        data, datalen, avg_denominator = self.align(data)
        # after aligning
        avgdata = []
        interlen = int(datalen / span)
        for i in range(0, interlen):
            tmp = 0
            for slave in self.slave:
                if len(data[slave][0]) == 2:
                    itmp = 0
                    for ii in range(0, span):
                        itmp += float(data[slave][i * span + ii][1])
                    itmp = itmp / span
                    tmp += itmp
                else:
                    mlist = len(data[slave])
                    for m in range(0, mlist):
                        itmp = 0
                        for ii in range(0, span):
                            itmp += float(data[slave][m][i * span + ii][1])
                        itmp = itmp / span
                        tmp += itmp

            tmp = tmp / avg_denominator
            if per:
                tmp = round(tmp * 100, 2)
            else:
                tmp = round(tmp, 2)
            avgdata.append([i, tmp])
        return avgdata

    def cal_std(self, data, span=1):
        """
        计算标准方差
        """
        data, datalen, avg_denominator = self.align(data)
        # after aligning
        avgdata = []
        interlen = int(datalen / span)
        for i in range(0, interlen):
            tmp = 0
            tostd = []
            for slave in self.slave:
                if len(data[slave][0]) == 2:
                    itmp = 0
                    for ii in range(0, span):
                        itmp += float(data[slave][i * span + ii][1])
                    itmp = itmp / span
                    tostd.append(itmp)
                else:
                    mlist = len(data[slave])
                    for m in range(0, mlist):
                        itmp = 0
                        for ii in range(0, span):
                            itmp += float(data[slave][m][i * span + ii][1])
                        itmp = itmp / span
                        tostd.append(itmp)

            tmp = np.std(tostd)
            avgdata.append([i, tmp])
        return avgdata

    def parse_cputil(self):
        """
        获取cpu利用率信息
        (1 - avg(irate(node_cpu_seconds_total{instance=~"$node",mode="idle"}[5m])) by (instance))*100
        """
        cpudata = {}
        for slave in self.slave:
            self.params['query'] = self.CPU_UTIL % slave
            r = requests.get(self.URL, params=self.params)
            if r.status_code == 200:
                results = json.loads(r.text)['data']['result']
                cpudata[slave] = results[0]['values']  # [[1,2][]]
                with open("%s/cpu.csv" % slave, "a", newline='') as f:
                    spamwriter = csv.writer(f)
                    spamwriter.writerows(cpudata[slave])

        avg_data = self.cal_avg(cpudata)
        std_data = self.cal_std(cpudata)
        self.grid(avg_data, "CPUUTIL_AVG")
        self.grid(std_data, "CPUUTIL_STD")

    def parse_mem(self):
        """
        获取mem信息
        node_memory_MemTotal_bytes{instance=~"$node"} - node_memory_MemAvailable_bytes{instance=~"$node"}
        """
        memdata = {}
        for slave in self.slave:
            self.params['query'] = self.MEM % (slave, slave, slave)
            r = requests.get(self.URL, params=self.params)
            if r.status_code == 200:
                results = json.loads(r.text)['data']['result']
                memdata[slave] = results[0]['values']
                with open("%s/mem.csv" % slave, "a", newline='') as f:
                    spamwriter = csv.writer(f)
                    spamwriter.writerows(memdata[slave])
        avg_data = self.cal_avg(memdata)
        std_data = self.cal_std(memdata)
        self.grid(avg_data, "MEMUTIL_AVG")
        self.grid(std_data, "MEMUTIL_STD")

    def parse_gpuutil(self):
        """
        获取gpu利用率信息
        DCGM_FI_DEV_GPU_UTIL
        获取到的格式为：[{"metric":{"UUID":**,"__name__":**,"gpu":**,"instance":**,"job":**},"values":[[time,value]]}]
        """
        self.params['query'] = self.GPU_UTIL
        r = requests.get(self.URL, params=self.params)
        gpuutildata = {}
        if r.status_code == 200:
            results = json.loads(r.text)['data']['result']

            for i,rs in enumerate(results):
                ip = rs["metric"]["instance"].split(":")[0]
                if ip not in gpuutildata:
                    gpuutildata[ip] = []
                gpuutildata[ip].append(rs["values"])
                if not os.path.exists(ip):
                    os.mkdir(ip)
                with open("%s/gpu%s.csv" % (ip, rs['metric']['gpu']), "a", newline='') as f:
                    spamwriter = csv.writer(f)
                    spamwriter.writerows(rs["values"])

        self.slave = list(map(lambda x: x.split(":")[0], list(gpuutildata.keys())))


        avg_data = self.cal_avg(gpuutildata)
        std_data = self.cal_std(gpuutildata)
        self.grid(avg_data, "GPU_AVG")
        self.grid(std_data, "GPU_STD")

    def parse_gpumem(self):
        """
        获取gpu-mem信息
        DCGM_FI_DEV_FB_USED/(DCGM_FI_DEV_FB_FREE+DCGM_FI_DEV_FB_USED)
        获取到的格式为：{"metric":{"UUID":**,"__name__":**,"gpu":**,"instance":**,"job":**},"values":[[time,value]]}
        """
        self.params['query'] = self.GPU_MEM
        r = requests.get(self.URL, params=self.params)
        gpumemdata = {}
        if r.status_code == 200:
            results = json.loads(r.text)['data']['result']
            for i, rs in enumerate(results):
                ip = rs["metric"]["instance"].split(":")[0]
                if ip not in gpumemdata:
                    gpumemdata[ip] = []
                gpumemdata[ip].append(rs["values"])
                if not os.path.exists(ip):
                    os.mkdir(ip)
                with open("%s/gpumem%s.csv" % (ip, rs['metric']['gpu']), "a", newline='') as f:
                    spamwriter = csv.writer(f)
                    spamwriter.writerows(rs["values"])

        avg_data = self.cal_avg(gpumemdata)
        std_data = self.cal_std(gpumemdata)
        self.grid(avg_data, "GPUMEMUTIL_AVG")
        self.grid(std_data, "GPUMEMUTIL_STD")

    def grid(self, data, type):
        time = []
        value = []
        for line in data:
            time.append(line[0])
            value.append(line[1])
        # search the line including accuracy
        print(time)
        time = [float(v) / 3600 for v in time]
        EVOL = [round(v, 2) for v in value]

        # 多条线画在一个图
        ax = plt.subplot(1, 1, 1)  # 注意:一般都在ax中设置,不再plot中设置
        # 设置x轴和y轴数据和属性
        plt.plot(time, EVOL, 'b', label='EVOL', linewidth='1.0', color='darkgoldenrod')

        # plt.title('Val.Acc/GPU Men./Util. VS Batch Size')
        plt.grid(axis="y")
        EVOLmean = np.array(EVOL).mean()

        font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': 12}
        plt.xlabel('Hours', font)
        plt.tick_params(labelsize=12)
        # plt.legend(['8 GPUs: average(6H-20H)='+str('%.2f%%'%EVOLmean),'16 GPUs: average(6H-20H)='+str('%.2f%%'%GSmean),'32 GPUs: average(6H-20H)='+str('%.2f%%'%RSmean),
        #             '64 GPUs: average(6H-20H)='+str('%.2f%%'%tpemean),'128 GPUs: average(6H-20H)='+str('%.2f%%'%nnnnmean)],loc='upper right',prop=font) # 显示图例
        plt.legend(['1 Node: average=' + str('%.2f' % EVOLmean + "%")], loc='upper right', prop=font)  # 显示图例

        # 设置刻度间隔，以及刻度格式
        # xmajorLocator = MultipleLocator(2)  # 将x主刻度标签设置为20的倍数
        # xmajorFormatter = FormatStrFormatter('%1d')  # 设置x轴标签文本的格式
        # xminorLocator   = MultipleLocator(50) #将x轴次刻度标签设置为5的倍数

        ymajorLocator = MultipleLocator(10)  # 0.5, 2, 5) #将y轴主刻度标签设置为0.5的倍数
        ymajorFormatter = FormatStrFormatter('%d')  # 1.1f') #设置y轴标签文本的格式
        plt.ylabel('{type}. Util.(%)'.format(type=type), font)
        # plt.ylim(1, 5.5)
        # plt.ylim(40, 90)
        # plt.ylim(0, 20)
        # plt.ylim(15, 50)

        # yminorLocator   = MultipleLocator(10) #将此y轴次刻度标签设置为0.1的倍数
        # ax.xaxis.set_major_locator(xmajorLocator)
        # ax.xaxis.set_major_formatter(xmajorFormatter)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_major_formatter(ymajorFormatter)

        # 设置x轴y轴范围
        # plt.xlim(0, 72000 / 3600)
        # plt.xlim(0, 72000 / 60)

        # 显示次刻度标签的位置,没有标签文本
        # ax.xaxis.set_minor_locator(xminorLocator)
        # ax.yaxis.set_minor_locator(yminorLocator)
        # 隐藏右侧顶部边框
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # 给线上的每个点加数值
        # for x, y in zip(time, GPU_Util):
        #     plt.text(x,y,y,ha='center',va='bottom',fontsize=9)
        # for x, y in zip(time, GPU_Mem):
        #     plt.text(x,y,y,ha='center',va='bottom',fontsize=9)
        # for x, y in zip(time, val_acc):
        #     plt.text(x,y,y,ha='center',va='bottom',fontsize=9)

        # ax.xaxis.grid(True, which='minor') #x坐标轴的网格使用主刻度
        # ax.yaxis.grid(True, which='major') #x坐标轴的网格使用主刻度
        ax.yaxis.grid(True, which='major', linestyle='--')  # y坐标轴的网格使用次刻度
        plt.savefig('%s.pdf' % type)
        plt.close()
        # plt.show()


if __name__ == "__main__":
    # gr = GenReport()
    # gr.output(pdf=True)
    gp = GenPerfdata()
    gp.parse_gpuutil()
    gp.parse_cputil()
    gp.parse_gpumem()
    gp.parse_mem()
    print("------------")
