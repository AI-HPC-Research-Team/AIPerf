![](https://github.com/AI-HPC-Research-Team/AIPerf/blob/master/logo.JPG)<br>

![](https://github.com/AI-HPC-Research-Team/AIPerf/blob/master/logo_PCL.jpg) ![](https://github.com/AI-HPC-Research-Team/AIPerf/blob/master/logo_THU.jpg)

### 总规划师：清华大学陈文光教授 ###
### 开发单位：鹏城实验室(PCL)，清华大学(THU)
### 特别感谢： 国防科技大学窦勇老师及其团队的宝贵意见，建议和支持 ###


- [AIPerf Benchmark v1.0](#head1)
	- [ Benchmark结构设计](#head2)
	- [ Benchmark安装说明](#head3)
		- [ 本文用于在容器环境下运行Benchmark](#head4)
		- [ 一、Benchmark环境配置、安装要求](#head5)
			- [ 1.物理机环境配置](#head6)
			- [ 2.容器制作](#head7)
			- [ 3.容器部署](#head8)
			- [ 4.数据集制作](#head9)
		- [ 二、Benchmark测试规范](#head10)
			- [ 配置运行参数](#head11)
			- [ 运行benchmark](#head12)
			- [ 停止实验](#head13)
		- [ 三、测试参数设置及推荐环境配置](#head14)
			- [ 可变设置](#head15)
			- [ 推荐环境配置](#head16)
	- [ Benchmark报告反馈](#head17)
	- [ 许可](#head18)



# <span id="head1">AIPerf Benchmark v1.0</span>

## <span id="head2"> Benchmark结构设计</span>

### 关于AIPerf设计理念，技术细节，以及测试结果，请参考论文：https://arxiv.org/abs/2008.07141 ###

AIPerf Benchmark基于微软NNI开源框架，以自动化机器学习（AutoML）为负载，使用network morphism进行网络结构搜索和TPE进行超参搜索。

Master节点将模型历史及其达到的正确率发送至Slave节点。Slave节点根据模型历史及其正确率，搜索出一个新的模型，并进行训练。Slave节点将根据某种策略（如连续10个Epoch的测试集正确率没有提升）停止训练，并将此模型及其达到的正确率发送至Master节点。Master节点接收并更新模型历史及正确率。
现有NNI框架在模型搜索阶段在Master节点进行，该特性是的AutoML作为基准测试程序负载时成为了发挥集群计算能力的瓶颈。为提升集群设备的计算资源利用率，项目组需要从减少Master节点计算时间、提升Slave节点GPU有效计算时间的角度出发，对AutoML框架进行修改。主要分以下特性：
将网络结构搜索过程分散到Slave节点上进行，有效利用集群资源优势；

1. 将每个任务的模型生成与训练过程由串行方式改为异步并行方式进行，在网络结构搜索的同时使得GPU可以同时进行训练，减少GPU空闲时间；
2. 将模型搜索过程中进行结构独特性计算部分设置为多个网络结构并行计算，减少时间复杂度中网络结构个数（n）的影响，可以以并发个数线性降低时间负载度；
3. 为从根本上解决后期模型搜索时需要遍历所有历史网络结构计算编辑距离的问题，需要探索网络结构独特性评估的优化算法或搜索效率更高的NAS算法，将其作为NAS负载添加至Benchmark框架中。

为进一步提升设备的利用率、完善任务调度的稳定性，修改、完善了调度代码，将网络结构搜索算法分布到每个slave节点执行，并采用slurm分配资源、分发任务。

Benchmark模块结构组成如下：

1. 源代码（AIPerf/src）：AIPerf主体模块为src模块，该模块包含了整个AIPerf主体框架

2. 参数初始化（AIPerf/examples/trials/network_morphism/imagenet/config.yml）：在AIPerf运行之前对参数进行调整

3. 日志&结果收集（AIPerf/scripts/reports）： 在AIPerf运行结束后将不同位置的日志和测试数据统一保存在同一目录下

4. 数据分析（AIPerf/scripts/reports）： 对正在运行/结束的测试进行数据分析，得出某一时间点内该测试的Error、Score、Regulated Score，并给出测试报告

5. 资源监控： 监控测试过程中的硬件资源使用，有助于测试分析和发现瓶颈 

	1. (必需)自动化脚本资源监控（AIPerf/examples/trials/network_morphism/imagenet/resource_monitor.py）
	2. (可选) 可视化资源监控（AIPerf/scripts/monitor）

***NOTE：后续文档的主要內容由Benchmark环境配置、安装要求，测试规范，报告反馈要求以及必要的参数设置要求组成；***

## <span id="head3"> Benchmark安装说明</span>

### <span id="head4"> 本文用于在容器环境下运行Benchmark</span>

如果用物理机环境测试，请移步: [物理机运行说明](https://github.com/AI-HPC-Research-Team/AIPerf/blob/master/README_OS.md)

### <span id="head5"> 一、Benchmark环境配置、安装要求</span>

*(本文档默认物理机环境已经安装docker、nvidia-docker)*

Benchmark运行环境由Master节点-Slaves节点组成，其中Mater节点不参与调度不需要配置GPU/加速卡，Slave节点可配置多块加速卡。

Benchmark运行时，需要先获取集群资源各节点信息（包括IP、环境变量等信息），根据各节点信息组建slurm调度环境，以master节点为slurm控制节点，各slave节点为slurm的计算节点。以用户的共享文件目录作为数据集、实验结果保存和中间结果缓存路径。
同时Master节点分别作为Benchmark框架和slurm的控制节点，根据实验配置文件中的最大任务数和slurm实际运行资源状态分配当前运行任务（trial）。每个trial分配至一个slave节点，trial的训练任务以节点中8GPU数据并行的方式执行训练。

#### <span id="head6"> 1.物理机环境配置</span>

(物理机执行：默认root用户操作)

**配置共享文件系统**

配置共享文件系统需要在物理机环境中进行，若集群环境中已有共享文件系统则跳过配置共享文件系统的步骤,若无共享文件系统，则需配置共享文件系统。

*搭建NFS*

AIPerf运行过程所有节点将使用NFS共享文件系统进行数据共享和存储

*安装NFS服务端*

将NFS服务端部署在master节点

```
apt install nfs-kernel-server -y
```

*配置共享目录*

创建共享目录/userhome，后面的所有数据共享将会在/userhome进行

```
mkdir /userhome
```

*修改权限*

```
chmod -R 777 /userhome
```

*打开NFS配置文件，配置NFS*

```
vim /etc/exports
```

添加以下内容

```
/userhome   *(rw,sync,insecure,no_root_squash)
```

*重启NFS服务*

```
service nfs-kernel-server restart
```

*安装NFS客户端*

所有slave节点安装NFS客户端

```
apt install nfs-common -y
```

slave节点创建本地挂载点

```
mkdir /userhome
```

slave节点将NFS服务器的共享目录挂载到本地挂载点/userhome

```
mount NFS-server-ip:/userhome /userhome
```

*检查NFS服务*

在任意节点执行

```
touch /userhome/test
```

如其他节点能在/userhome下看见 test 文件则运行正常。

**物理机下载基础镜像**

针对NVIDIA V100
```
docker pull nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
```
针对NVIDIA A100
```
docker pull nvidia/cuda:11.1-cudnn8-devel-ubuntu16.04
```

**启动容器**

针对NVIDIA V100
```
nvidia-docker run -it --name build_AIPerf -v /userhome:/userhome nvidia/cuda:10.1-cudnn7-devel-ubuntu16.04
```
针对NVIDIA A100
```
nvidia-docker run -it --name build_AIPerf -v /userhome:/userhome nvidia/cuda:11.1-cudnn8-devel-ubuntu16.04
```

#### <span id="head7"> 2.容器制作</span>

(容器内执行)

**安装基础工具**

```
apt update && apt install git vim cmake make openssh-client openssh-server wget tzdata  curl sshpass -y
```

*配置ssh-server*

开启ssh root登录权限,修改ssh配置文件 /etc/ssh/sshd_config

```
vim /etc/ssh/sshd_config
```

找到PermitRootLogin prohibit-password所在行，并修改为

```
#PermitRootLogin prohibit-password
PermitRootLogin yes
```

避免和物理机端口冲突，打开配置文件 /etc/ssh/sshd_config，修改ssh端口22为222

```
#Port 22
port 222
```

*为root用户设置密码*

```
passwd
```

密码设置为123123

*配置时区*

```
dpkg-reconfigure tzdata
```

选择Asia -> Shanghai

*配置中文支持和环境变量*

在/etc/bash.bashrc最后添加

```
export LANG=C.UTF-8
export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
```

**配置python运行环境**

*安装python3.5*

```
apt install --install-recommends python3 python3-dev python3-pip -y
```

*升级pip*

```
pip3 install --upgrade pip
```

**安装AIPerf**

*下载源代码到共享目录/userhome*

```shell
git clone https://github.com/AI-HPC-Research-Team/AIPerf.git /userhome/AIPerf
```

*安装python环境库*

```
cd /userhome/AIPerf
pip3 install -r requirements.txt --timeout 3000
```

*编译安装*

```
source install.sh
```

*检查AIPerf安装*

执行

```
nnictl --help
```

如果打印帮助信息，则安装正常

**安装slurm**

AIPerf的资源调度通过slurm进行

*安装slurm、munge*

```
apt install munge slurm-llnl -y
```

*创建munge秘钥*

```
/usr/sbin/create-munge-key -r
```

**目录调整**

*创建必要的目录*

mountdir 存放实验过程数据，nni存放实验过程日志

```shell
mkdir /userhome/mountdir
mkdir /userhome/nni
```

将共享目录下的相关目录链接到用户home目录下

```shell
ln -s /userhome/mountdir /root/mountdir
ln -s /userhome/nni /root/nni
```

*必要的路径及数据配置*

 将权重文件复制到共享目录/userhome中

```shell
wget -P /userhome https://github.com/fchollet/deep-learning-models/releases/download/v0.1/resnet50_weights_tf_dim_ordering_tf_kernels.h5
```


#### <span id="head8"> 3.容器部署</span>

(物理机执行)

**提交容器为镜像**

```
sudo docker commit build_AIPerf aiperf:latest
```

**导出镜像**

将容器导出到之前创建好的共享目录/userhome，方便其它节点导入

```
sudo docker save -o  /userhome/AIPerf.tar aiperf:latest
```

**导入镜像**

参与实验的所有节点导入镜像，由于镜像需要通过NFS传输到其他节点，需要一些时间

```
sudo docker load -i /userhome/AIPerf.tar
```

**运行容器**

参与实验的所有节点运行容器，master节点可以用docker运行容器，不需要用 nvidia-docker

```
sudo nvidia-docker run -it --net=host -v /userhome:/userhome aiperf:latest
```

**配置容器**

(容器内操作)

*所有节点容器重启ssh服务*

```
service ssh restart
```

*配置slurm*

以下操作在master节点进行，slurm将获取所有slave节点中cpu核数最低的节点的核数，并将该核数配置为每个slave节点的最高可用核数，而并非每个节点各自的实际核数。

进入/userhome/AIPerf/scripts/autoconfig_slurm目录

```
cd /userhome/AIPerf/scripts/autoconfig_slurm
```

*进行ip地址配置*

1. 将所有slave节点ip按行写入slaveip.txt。
2. 将master节点ip写入masterip.txt。
3. 确保所有节点的ssh用户、密码、端口是一致的，并根据自身情况修改 slurm_autoconfig.sh脚本中的用户名和密码。

*运行自动配置脚本*

```
bash slurm_autoconfig.sh
```

slurm配置完成后会提示当前所有节点最高可用核数并给出后续config.yml中slurm的运行参数`srun --cpus-per-task=xx`

*检查slurm*

执行命令查看所有节点状态

```
sinfo
```

如果所有节点STATE列为idle则slurm配置正确，运行正常。

如果STATE列为unk，等待一会再执行sinfo查看，如果都为idle，则slurm配置正确，运行正常。

如果STATE列的状态后面带*则该节点网络出现问题master无法访问到该节点。

**启动prometheus&grafana(可选)**

*slave节点启动资源监控服务*

进入到先前安装指定的路径执行

```
./node_exporter &
dcgm-exporter &
```

*master节点执行安装脚本*

1）在master节点执行脚本monitor_master_run.sh

```
cd  /userhome/AIPerf/scripts/monitor
bash monitor_master_run.sh -i 安装路径  -ip slaveip.txt的绝对路径
```

*访问grafana查看资源信息*

打开浏览器访问 master_ip:3000,初始账号密码为admin/admin;

*grafana增加数据源*

在左侧菜单栏按顺序点击以下按钮

configuration ->Data Sources

点击Prometheus,在URL框中填入master_ip:9090，点击 Save & Test 按钮

*导入模板文件*

在左侧菜单栏按顺序点击以下按钮

Create -> Import

点击 Upload .json file 导入 'AIPerf/scripts/monitor/monitor.json'

点击 load 即可看到监控的资源使用情况

**其他操作**

*重启资源监控*

当机器重启后监控服务会被关闭，需要手动启动

*master重启服务*

进入到先前安装指定的路径执行

```
./prometheus &
service grafana-server restart
```

*slave重启服务*

进入到先前安装指定的路径执行

```
./node_exporter &
dcgm-exporter &
```

#### <span id="head9"> 4.数据集制作</span>

制作数据集建议在已做好容器内操作，里面包含了制作数据集需要的基本环境。

**数据集下载**

 *Imagenet官方地址：http://www.image-net.org/index* 

官方提供四种数据集：  Flowers、CIFAR-10、MNIST、ImageNet-2012  前三个数据集数据量小，直接调用相关脚本自动会完成下载、转换（TFRecord格式）的过程，在 /userhome/AIPerf/scripts/build_data目录下执行以下脚本：

```javascript
cd  /userhome/AIPerf/scripts/build_data
./download_imagenet.sh
```

原始的ImageNet-2012下载到当前的imagenet目录并包含以下两个文件:

- ILSVRC2012_img_val.tar
- ILSVRC2012_img_train.tar

**TFReord制作**

训练集和验证集需要按照1000个子目录下包含图片的格式，处理步骤：

1. 将train 和 val 的数据按照文件夹分类
2. 指定参数运行build_imagenet_data.py

**可以按照以下步骤执行**:  假设数据存放在/userhome/AIPerf/scripts/build_data/imagenet目录下，TFRecord文件的输出目录是/userhome/AIPerf/scripts/build_data/ILSVRC2012/output

```shell
# 做验证集
cd  /userhome/AIPerf/scripts/build_data
mkdir -p ILSVRC2012/raw-data/imagenet-data/validation/  
tar -xvf imagenet/ILSVRC2012_img_val.tar -C ILSVRC2012/raw-data/imagenet-data/validation/
python preprocess_imagenet_validation_data.py ILSVRC2012/raw-data/imagenet-data/validation/ imagenet_2012_validation_synset_labels.txt

# 做训练集
mkdir -p ILSVRC2012/raw-data/imagenet-data/train/
tar -xvf imagenet/ILSVRC2012_img_train.tar -C ILSVRC2012/raw-data/imagenet-data/train/ && cd ILSVRC2012/raw-data/imagenet-data/train
find . -name "*.tar" | while read NAE ; do mkdir -p "${NAE%.tar}"; tar -xvf "${NAE}" -C "${NAE%.tar}"; rm -f "${NAE}"; done
cd -

# 执行转换
mkdir -p ILSVRC2012/output
python build_imagenet_data.py --train_directory=ILSVRC2012/raw-data/imagenet-data/train --validation_directory=ILSVRC2012/raw-data/imagenet-data/validation --output_directory=ILSVRC2012/output --imagenet_metadata_file=imagenet_metadata.txt --labels_file=imagenet_lsvrc_2015_synsets.txt
```

上面步骤执行完后，路径ILSVRC2012/output包含128个validation开头的验证集文件和1024个train开头的训练集文件。需要分别将验证集和数据集移动到共享目录/root/datasets/imagenet对应路径下

```
mkdir -p /root/datasets/imagenet/train
mkdir -p /root/datasets/imagenet/val
mv ILSVRC2012/output/train-* /root/datasets/imagenet/train
mv ILSVRC2012/output/validation-* /root/datasets/imagenet/val
```

### <span id="head10"> 二、Benchmark测试规范</span>

为了使结果有效，测试满足的基本条件是：
1. 测试运行时间应不少于1小时；
2. 测试的计算精度不低于FP-16；
3. 测试完成时所取得的最高正确率应大于70%；

#### <span id="head11"> 初始化配置</span>

*(以下操作均在master节点进行)*
根据需求修改/userhome/AIPerf/example/trials/network_morphism/imagenet/config.yml配置

|      |         可选参数         |              说明               |     默认值      |
| ---- | :----------------------: | :-----------------------------: | :-------------: |
| 1    |     trialConcurrency     |        同时运行的trial数        |        1        |
| 2    |     maxExecDuration      |     设置测试时间(单位 ：h)      |       12        |
| 3    |   CUDA_VISIBLE_DEVICES   |    指定测试程序可用的gpu索引    | 0,1,2,3,4,5,6,7 |
| 4    | srun：--cpus-per-task=30 |   参数为slurm可用cpu核数减 1    |       30        |
| 5    |         --slave          | 跟 trialConcurrency参数保持一致 |        1        |
| 6    |           --ip           |          master节点ip           |    127.0.0.1    |
| 7    |       --batch_size       |           batch size            |       448       |
| 8    |         --epochs         |         正常训练epoch数         |       60        |
| 9    |       --initial_lr       |           初始学习率            |      1e-1       |
| 10   |        --final_lr        |           最终学习率            |        0        |
| 11   |     --train_data_dir     |         训练数据集路径          |      None       |
| 12   |      --val_data_dir      |         验证数据集路径          |      None       |
| 13   |        --warmup_1        |    warm up机制第一轮epoch数     |       15        |
| 14   |        --warmup_2        |    warm up机制第二轮epoch数     |       30        |
| 15   |        --warmup_3        |    warm up机制第三轮epoch数     |       45        |
| 16   |   --num_parallel_calls   |      tfrecord数据加载加速       |       48        |

可参照如下配置：

```
authorName: default
experimentName: example_imagenet-network-morphism-test
trialConcurrency: 1		# 1
maxExecDuration: 12h	# 2
maxTrialNum: 30000
trainingServicePlatform: local
useAnnotation: false
tuner:
 \#choice: TPE, Random, Anneal, Evolution, BatchTuner, NetworkMorphism
 \#SMAC (SMAC should be installed through nnictl)
 builtinTunerName: NetworkMorphism
 classArgs:
  optimize_mode: maximize
  task: cv
  input_width: 224
  input_channel: 3
  n_output_node: 1000
  
trial:
 command: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  \                                  # 3
       srun -N 1 -n 1 --ntasks-per-node=1 \
       --cpus-per-task=30 \	  # 4
       python3 imagenet_tfkeras_slurm_hpo.py \
       --slave 1 \								  # 5
       --ip 127.0.0.1 \							  # 6
       --batch_size 448 \						  # 7
       --epoch 60 \						          # 8
       --initial_lr 1e-1 \						  # 9
       --final_lr 0 \						  # 10
       --train_data_dir /gdata/ILSVRC2012/ImageNet-Tensorflow/train_tfrecord/ \  # 11
       --val_data_dir /gdata/ILSVRC2012/ImageNet-Tensorflow/validation_tfrecord/ # 12

 codeDir: .
 gpuNum: 0
```

#### <span id="head12"> 运行benchmark</span>

在/userhome/AIPerf/example/trials/network_morphism/imagenet/目录下执行以下命令运行用例

```
nnictl create -c config.yml
```

**查看运行过程**

执行以下命令查看正在运行的experiment的trial运行信息

```
nnictl top
```

当测试运行过程中，运行以下程序会在终端打印experiment的Error、Score、Regulated Score等信息

```
python3 /userhome/AIPerf/scripts/reports/report.py --id  experiment_ID  
```

#### <span id="head13"> 停止实验</span>

停止expriments, 执行

```
nnictl stop
```

通过命令squeue查看slurm中是否还有未被停止的job，如果存在job且ST列为CG，请等待作业结束，实验才算完全停止。

**查看实验报告**

当测试运行过程中，运行以下程序会在终端打印experiment的Error、Score、Regulated Score等信息

```
python3 /userhome/AIPerf/scripts/reports/report.py --id  experiment_ID  
```

同时会产生实验报告存放在experiment_ID的对应路径/userhome/mountdir/nni/experiments/experiment_ID/results目录下

实验成功时报告为 Report_Succeed.html

实验失败时报告为 Report_Failed.html

实验失败会报告失败原因，请查阅AI Benchmark测试规范分析失败原因

**保存日志&结果数据**

运行以下程序可将测试产生的日志以及数据统一保存到/userhome/mountdir/nni/experiments/experiment_ID/results/logs中，便于实验分析

```
python3 /userhome/AIPerf/scripts/reports/report.py --id  experiment_ID  --logs True
```

由于实验数据在复制过程中会导致额外的网络、内存、cpu等资源开销，建议在实验停止/结束后再执行日志保存操作。



### <span id="head14"> 三、测试参数设置及推荐环境配置</span>

#### <span id="head15"> 可变设置</span>

1. slave计算节点的GPU卡数：默认将单个物理服务器作为一个slave节点，并使用其所有GPU；
2. 深度学习框架：默认使用keras+tensorflow；
3. 数据集加载方式：默认将数据预处理成TFRecord格式，以加快数据加载的效率；
4. 数据集存储方式：默认采用网络共享存储；
5. 超参设置：默认初始batch size=256，默认初始学习率=0.1，默认最终学习率=0，默认正常训练epochs=60，默认从第四轮trial开始，每个trial搜索1次，默认超参为kernel size和batch size。

#### <span id="head16"> 推荐环境配置</span>

- 环境：Ubuntu16.04，docker=18.09.9，SLURM=v20.02

- 软件：TensorFlow2.2.0，CUDA10.1，python3.5
- Container：36个物理CPU核，512GB内存，8张GPU


***NOTE: 推荐基于Intel Xeon Skylake Platinum8268 and NVIDIA Tesla NVLink v100配置***





## <span id="head17"> Benchmark报告反馈</span>

若测试中遇到问题，请联系renzhx@pcl.ac.cn，并附上`/userhome/mountdir/nni/experiments/experiment_ID/results/`中的html版报告。

## <span id="head18"> 许可</span>

基于 MIT license
