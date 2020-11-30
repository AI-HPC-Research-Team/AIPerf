![](https://github.com/AI-HPC-Research-Team/AIPerf/blob/Atlas-800/logo.JPG)<br>

![](https://github.com/AI-HPC-Research-Team/AIPerf/blob/Atlas-800/logo_PCL.jpg) ![](https://github.com/AI-HPC-Research-Team/AIPerf/blob/Atlas-800/logo_THU.jpg)
### 开发单位：鹏城实验室(PCL)，清华大学(THU)
### 特别感谢国防科技大学窦勇老师及其团队的宝贵意见和支持 ###


- [AIPerf Benchmark v1.0](#head1)
	- [ Benchmark结构设计](#head2)
	- [ Benchmark安装说明](#head3)
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

AIPerf Benchmark基于微软NNI开源框架，以自动化机器学习（AutoML）为负载，使用network morphism进行网络结构搜索，利用TPE进行超参搜索。

Master节点将模型历史及其达到的正确率发送至Slave节点，Slave节点根据模型历史及其正确率，搜索出一个新的模型，并进行训练。Slave节点将根据某种策略停止训练，并将此模型及其达到的正确率发送至Master节点。Master节点接收并更新模型历史及正确率。

现有NNI框架的模型搜索阶段在Master节点进行，该特性是的AutoML作为基准测试程序负载时成为了发挥集群计算能力的瓶颈。为提升集群设备的计算资源利用率，项目组需要从减少Master节点计算时间、提升Slave节点NPU有效计算时间的角度出发，对AutoML框架进行修改。主要分以下特性：
将网络结构搜索过程分散到Slave节点上进行，有效利用集群资源优势；

1. 将每个任务的模型生成与训练过程由串行方式改为异步并行方式进行，在网络结构搜索的同时使得Ascend910可以同时进行训练，减少Ascend910空闲时间；
2. 将模型搜索过程中进行结构独特性计算部分设置为多个网络结构并行计算，减少时间复杂度中网络结构个数（n）的影响，可以以并发个数线性降低时间负载度；
3. 为从根本上解决后期模型搜索时需要遍历所有历史网络结构计算编辑距离的问题，需要探索网络结构独特性评估的优化算法或搜索效率更高的NAS算法，将其作为NAS负载添加至Benchmark框架中。

为进一步提升设备的利用率、完善任务调度的稳定性，修改、完善了调度代码，将网络结构搜索算法分布到每个slave节点执行，并采用slurm分配资源、分发任务。

Benchmark模块结构组成如下：

1. 源代码（AIPerf/src）：AIPerf主体模块为src模块，该模块包含了整个AIPerf主体框架

2. 参数初始化（AIPerf/examples/trials/network_morphism/imagenet/config.yml）：在AIPerf运行之前对参数进行调整

3. 日志&结果收集（AIPerf/scripts/reports）： 在AIPerf运行结束后将不同位置的日志和测试数据统一保存在同一目录下

4. 数据分析（AIPerf/scripts/reports）： 对正在运行/结束的测试进行数据分析，得出某一时间点内该测试的Error、Score、Regulated Score，并给出测试报告


***NOTE：后续文档的主要內容由Benchmark环境配置、安装要求，测试规范，报告反馈要求以及必要的参数设置要求组成；***

## <span id="head3"> Benchmark安装说明</span>

### <span id="head5"> 一、Benchmark环境配置、安装要求</span>

*(本文档默认物理机环境已经安装docker, 物理机系统与驱动安装详见 **Atlas800系统环境安装配置说明.md**文档)*

Benchmark运行环境由Master节点-Slaves节点组成，其中Mater节点需要同Slave节点保持相同的环境配置。

Benchmark运行时，需要先获取集群资源各节点信息（包括IP、环境变量等信息），根据各节点信息组建slurm调度环境，以master节点为slurm控制节点，各slave节点为slurm的计算节点。以用户的共享文件目录作为数据集、实验结果保存和中间结果缓存路径。

同时Master节点分别作为Benchmark框架和slurm的控制节点，根据实验配置文件中的最大任务数和slurm实际运行资源状态分配当前运行任务（trial）。每个trial分配至一个slave节点，trial的训练任务以节点中8张Ascend910加速卡数据并行的方式执行训练。

#### 1、物理机环境配置

当Master节点和Slave节点处于不同的物理机环境中时，需要在这些**物理机**中配置共享文件系统，方便所有节点能够共享同一种环境资源。如果Master节点和Slave节点处于同一台物理机，则可跳过此共享文件系统配置环节。

AIPerf 运行过程所有节点将使用 NFS 共享文件系统进行数据共享和存储，NFS 的搭建过程默认 root 用户在物理机中执行！

**(1) master 节点安装 NFS 服务端**

将 NFS 服务端安装在 master 节点：

```
apt install nfs-kernel-server -y
```

然后在 master 节点创建共享目录 `/userhome`，后面的所有数据共享将会在 `/userhome` 进行：

```
mkdir /userhome
chmod -R 777 /userhome
```

打开 NFS 配置文件，配置NFS：

```
vim /etc/exports
```

添加以下内容：

```
/userhome   *(rw,sync,insecure,no_root_squash)
```

重启 NFS 服务：

```
service nfs-kernel-server restart
```

**(2) 在 slave 节点安装 NFS 客户端**

所有 slave 节点安装 NFS 客户端

```
apt install nfs-common -y
```

然后在 slave 节点创建本地挂载点：

```
mkdir /userhome
```

每一个 slave 节点需要将 NFS 服务器的共享目录挂载到本地挂载点 `/userhome`，其中**NFS-server-ip**指第一步master节点ip：

```
mount NFS-server-ip:/userhome /userhome
```

**(3) 检查 NFS 服务**

在任意节点执行：

```
touch /userhome/test
```

如其他节点能在 `/userhome` 下看见 test 文件则表明 NFS 运行正常。

#### 2、镜像制作

AIPerf-atlas800 中的容器以 mindspore 作为基本的深度学习框架，因此需要联系华为 mindspore 开发人员获取 相关镜像，在此镜像基础上做如下的修改。

**注：基础镜像ubuntu_arm.tar由华为提供。**

**（1）创建基础容器**

```shell
docker load -i ubuntu_arm.tar  # load基础镜像
docker run --privileged -d -v /userhome:/userhome -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ --name build_AIPerf ubuntu_arm:r0.5 bash -c "service ssh restart; while true; do echo hello world; sleep 1;done"
```

进入容器的方式为：

```shell
docker exec -it build_AIPerf bash
```

**（2）容器内安装基础工具**

```shell
apt update && apt install git vim cmake make openssh-client openssh-server wget tzdata  curl sshpass libfreetype6-dev pkg-config python -y
```

开启 ssh root 登录权限，修改ssh配置文件 /etc/ssh/sshd_config：

```
vim /etc/ssh/sshd_config
```

找到PermitRootLogin prohibit-password所在行，并修改为：

```
#PermitRootLogin prohibit-password
PermitRootLogin yes
```

避免和物理机端口冲突，打开配置文件 /etc/ssh/sshd_config，修改ssh端口22为222：

```
#Port 22
port 222
```

为 root 用户设置密码：

```
passwd
```

密码设置为123123。配置时区：

```
dpkg-reconfigure tzdata
```

选择Asia -> Shanghai。配置中文支持，方法是在 /etc/bash.bashrc 最后添加：

```
export LANG=C.UTF-8
```

**（3）配置python运行环境**

镜像已经预装python3.7.5环境，如果没有请安装python3.7.5。然后添加路径到环境变量，方法是在 /etc/bash.bashrc 文件最后一行添加：

```
export PATH="/usr/local/python375/bin:$PATH"
```

最后，升级pip：

```
pip3 install --upgrade pip
```

**（4）安装AIPerf**

下载源代码到共享目录 /userhome：

```shell
git clone -b Atlas https://github.com/AI-HPC-Research-Team/AIPerf.git \
/userhome/AIPerf
```

安装python环境库：

```
cd /userhome/AIPerf
pip3 install -r requirements.txt --timeout 3000
```

编译安装：

```
source install.sh
```

检查AIPerf安装，执行：

```
nnictl --help
```

如果打印帮助信息，则安装正常

**注：此步骤耗时较长。** 

**（5）安装slurm**

AIPerf的资源调度通过slurm进行，在容器内安装slurm、munge：

```
apt install munge slurm-wlm slurm-wlm-basic-plugins -y
```

创建munge秘钥：

```
/usr/sbin/create-munge-key -r
```

**（6）目录调整**

mountdir 存放实验过程数据，nni存放实验过程日志，二者均是必要的日志目录：

```shell
mkdir /userhome/mountdir
mkdir /userhome/nni
```

将共享目录下的相关目录链接到用户home目录下：

```shell
ln -s /userhome/mountdir /root/mountdir
ln -s /userhome/nni /root/nni
```

在 `/userhome` 目录下创建 Ascend910 的环境变量配置文件 `docker_env.sh`，文件内容如下：

```shell
export SLOG_PRINT_TO_STDOUT=2
export GLOG_v=2

LOCAL_ASCEND=/usr/local/Ascend/nnae/20.0.0.B035/arm64-linux_gcc7.3.0/
export ASCEND_OPP_PATH=${LOCAL_ASCEND}/opp
export LD_LIBRARY_PATH=/usr/local/Ascend/add-ons/:${LOCAL_ASCEND}/fwkacllib/lib64/:${LD_LIBRARY_PATH}
export ME_TBE_PLUGIN_PATH=${LOCAL_ASCEND}/opp/framework/built-in/tensorflow
export TBE_IMPL_PATH=${LOCAL_ASCEND}/opp/op_impl/built-in/ai_core/tbe/
export PYTHONPATH=$TBE_IMPL_PATH:$PYTHONPATH
export PATH=${LOCAL_ASCEND}/fwkacllib/ccec_compiler/bin/:${PATH}
```

在 `/userhome` 目录下创建 mindspore 八卡分布式训练配置文件 `rank_table_8pcs.json`，文件内容如下：

```shell
{
    "board_id": "0x0020",
    "chip_info": "910",
    "deploy_mode": "lab",
    "group_count": "1",
    "group_list": [
        {
            "device_num": "8",
            "server_num": "1",
            "group_name": "",
            "instance_count": "8",
            "instance_list": [
                {"devices": [{"device_id": "0","device_ip": "192.168.60.2"}],"rank_id": "0","server_id": "172.17.0.3"},
                {"devices": [{"device_id": "1","device_ip": "192.168.61.2"}],"rank_id": "1","server_id": "172.17.0.3"},
                {"devices": [{"device_id": "2","device_ip": "192.168.62.2"}],"rank_id": "2","server_id": "172.17.0.3"},
                {"devices": [{"device_id": "3","device_ip": "192.168.63.2"}],"rank_id": "3","server_id": "172.17.0.3"},
                {"devices": [{"device_id": "4","device_ip": "192.168.60.3"}],"rank_id": "4","server_id": "172.17.0.3"},
                {"devices": [{"device_id": "5","device_ip": "192.168.61.3"}],"rank_id": "5","server_id": "172.17.0.3"},
                {"devices": [{"device_id": "6","device_ip": "192.168.62.3"}],"rank_id": "6","server_id": "172.17.0.3"},
                {"devices": [{"device_id": "7","device_ip": "192.168.63.3"}],"rank_id": "7","server_id": "172.17.0.3"}
                ]
        }
    ],
    "para_plane_nic_location": "device",
    "para_plane_nic_name": ["eth0","eth1","eth2","eth3","eth4","eth5","eth6","eth7"],
    "para_plane_nic_num": "8",
    "status": "completed"
}

```


至此，`/userhome` 目录下的文件结构如下，之后所有节点将通过 NFS 共享此目录：

```shell
|-- userhome
    |-- AIPerf						# 核心代码
    |-- docker_env.sh				# Ascend910 环境变量配置文件
    |-- rank_table_8pcs.json		# mindspore 多卡分布式训练配置文件
    |-- nni							# 实验信息输出目录
    |-- mountdir					# 实验结果输出目录，包括算分结果
```

**（7）镜像制作**

首先，退出容器，然后将上述容器提交为镜像：

```
sudo docker commit build_AIPerf aiperf:atlas
```

然后将镜像导出到之前创建好的共享目录 `/userhome`，方便其它节点导入：

```
sudo docker save -o  /userhome/AIPerf.tar aiperf:atlas
```



#### 3、数据集制作

Ascend910 使用华为开发的 mindspore 作为深度学习框架，训练使用 imagenet 原始数据集。

**（1）数据集下载**

Imagenet官方地址：http://www.image-net.org/index

在 /userhome/AIPerf/scripts/build_data 目录下执行以下脚本：

```javascript
cd ~
bash /userhome/AIPerf/scripts/build_data/download_imagenet.sh
```

原始的ImageNet-2012下载到当前的imagenet目录并包含以下两个文件:

- ILSVRC2012_img_val.tar
- ILSVRC2012_img_train.tar

**（2）解压数据**

训练集和验证集需要按照1000个子目录下包含图片的格式，处理步骤：

1. 解压压缩包
2. 将train 和 val 的数据按照文件夹分类

**可以按照以下步骤执行**:  假设数据存放在~/目录下，最终文件的输出目录是~/data

```shell
# 解压验证集
cd  ~
mkdir -p data/val
tar -xvf ILSVRC2012_img_val.tar -C ~/data/val
python3 /userhome/AIPerf/scripts/build_data/preprocess_imagenet_validation_data.py ~/data/val /userhome/AIPerf/scripts/build_data/imagenet_2012_validation_synset_labels.txt

# 解压训练集
cd ~
mkdir -p data/train
tar -xvf ILSVRC2012_img_train.tar -C ~/data/train && cd ~/data/train
find . -name "*.tar" | while read NAE ; do mkdir -p "${NAE%.tar}"; tar -xvf "${NAE}" -C "${NAE%.tar}"; rm -f "${NAE}"; done
```

上面步骤执行完后，路径~/data/下，val包含1000个训练集目录（共50000张验证集图片），train包含1000个训练集目录（共~1281167张验证集图片）。

***注：完成数据集制作后，数据集需拷贝至每个节点的相同路径下。***



#### 4、容器部署

在各节点部署第3步所制作的镜像并 创建相同的容器。首先，参与实验的所有节点需要导入镜像，由于镜像是通过NFS传输到其他节点的，所以需要一些时间：

```
sudo docker load -i /userhome/AIPerf.tar
```

容器创建方式如下（**保证NFS共享成功，并保证物理机内无其它正在运行的无关容器**）：

```
docker run --privileged -d --net=host -v /userhome:/userhome -v ~/data:/home/data -v /usr/local/Ascend/driver/:/usr/local/Ascend/driver/  -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/ --name build_test aiperf:atlas bash -c "service ssh restart; while true; do echo hello world; sleep 1;done"
```

**所有节点**在**进入容器**：

```shell
docker exec -it build_test bash
```

之后进入**master 节点**所在容器，进入 `/userhome/AIPerf/scripts/autoconfig_slurm` 目录执行以下操作：

1. 将所有 slave 节点ip按行写入 slaveip.txt；
2. 将 master 节点ip写入 masterip.txt；
3. 确保所有节点的ssh用户、密码、端口是一致的，并根据自身情况修改 slurm_autoconfig.sh 脚本中的用户名和密码；

重启ssh服务，然后运行自动配置脚本：

```shell
service ssh restart
bash slurm_autoconfig.sh
```

slurm配置完成后会提示当前所有节点最高可用核数并给出后续config.yml中slurm的运行参数`srun --cpus-per-task=xx`。执行命令查看所有节点状态：

```shell
sinfo
```

如果所有节点STATE列为 idle 则表示 slurm 配置正确，运行正常；如果STATE列为unk，等待一会再执行sinfo查看，如果都为idle，则slurm配置正确，运行正常；如果STATE列的状态后面带*则该节点网络出现问题，master无法访问到该节点。



#### 5、AIperf在集群上的自动部署（可选）

操作流程与部署脚本详见 <u>**./scripts/deploy/**</u> 内的说明文件。



### <span id="head10"> 二、Benchmark测试规范</span>

为了使结果有效，测试满足的基本条件是：
1. 测试运行时间应不少于1小时；
2. 测试的计算精度不低于FP-16；
3. 测试完成时所取得的最高正确率应大于70%；

#### 1、初始化配置

(以下操作均在master节点进行)

根据需求修改 /userhome/AIPerf/example/trials/network_morphism/imagenet/config.yml 配置

|      |         可选参数          |              说明               |  默认值   |
| ---- | :-----------------------: | :-----------------------------: | :-------: |
| 1    |     trialConcurrency      |        同时运行的trial数        |     1     |
| 2    |      maxExecDuration      |     设置测试时间(单位 ：h)      |    12     |
| 3    |          NPU_NUM          |  指定测试程序可用的加速卡数量   |     8     |
| 4    | srun：--cpus-per-task=191 |   参数为slurm可用cpu核数减 1    |    191    |
| 5    |          --slave          | 跟 trialConcurrency参数保持一致 |     1     |
| 6    |           --ip            |          master节点ip           | 127.0.0.1 |
| 7    |       --batch_size        |           batch size            |    256    |
| 8    |          --epochs         |         正常训练epoch数         |    60     |
| 9    |       --initial_lr        |           初始学习率            |   1e-1    |
| 10   |        --final_lr         |           最终学习率            |     0     |
| 11   |     --train_data_dir      |         训练数据集路径          |   None    |
| 12   |      --val_data_dir       |         验证数据集路径          |   None    |
| 13   |      --warmup_1        |    warm up机制第一轮epoch数     |       15        |
| 14   |      --warmup_2        |    warm up机制第二轮epoch数     |       30        |
| 15   |      --warmup_3        |    warm up机制第三轮epoch数     |       45        |


可参照如下配置：

```shell
authorName: default
experimentName: example_imagenet-network-morphism-test
trialConcurrency: 1		# 1
maxExecDuration: 12h	# 2
maxTrialNum: 6000
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
     command: NPU_NUM=8  \							# 3
       srun -N 1 -n 1 --ntasks-per-node=1 \
       --cpus-per-task=191 \						# 4
       python3 imagenet_train.py \
       --slave 1 \									# 5
       --ip 127.0.0.1 \								# 6
       --batch_size 256 \							# 7
       --epochs 60 \									# 8
       --initial_lr 1e-1 \							# 9
       --final_lr 0 \								# 10
       --train_data_dir /home/data/train/ \  # 11
       --val_data_dir /home/data/val/ \ # 12
       --warmup_1 15 \   # 13
       --warmup_2 30 \   # 14
       --warmup_3 45     # 15

 codeDir: .
 gpuNum: 0
```

#### 2、运行benchmark

在**所有节点**的容器内，先导入环境变量，并执行以下命令运行用例

```shell
source /userhome/docker_env.sh
```

在**master节点**容器内，执行以下命令运行用例

*注：运行用例前确认NFS是否挂载成功*

```shell
cd /userhome/AIPerf/examples/trials/network_morphism/imagenet/
nnictl create -c config.yml
```

在**master节点**容器内，执行以下命令查看正在运行的experiment的trial运行信息

```
nnictl top
```

当测试运行过程中，运行以下程序会在终端打印experiment的Error、Score、Regulated Score等信息，experiment_ID 代表当前实验ID

```
python3 /userhome/AIPerf/scripts/reports/report.py --id experiment_ID  
```

#### 3、停止实验

在**master节点**容器内，停止experiments，执行：

```
nnictl stop
```

通过命令squeue查看slurm中是否还有未被停止的job，如果存在job且ST列为CG，请等待作业结束，实验才算完全停止。

实验整体结束后中，运行以下程序会在终端打印experiment的Error、Score、Regulated Score等信息：

```
python3 /userhome/AIPerf/scripts/reports/report.py --id  experiment_ID  
```

同时会产生实验报告存放在experiment_ID的对应路径/root/mountdir/nni/experiments/experiment_ID/results目录下。实验成功时报告为 Report_Succeed.html；实验失败时报告为 Report_Failed.html；实验失败会报告失败原因，请查阅AI Benchmark测试规范分析失败原因。

运行以下程序，可将实验产生的日志以及数据统一保存到 `/userhome/mountdir/nni/experiments/experiment_ID/results/logs` 中，便于实验分析

```
python3 /userhome/AIPerf/scripts/reports/report.py --id  experiment_ID  --logs True
```

由于实验数据在复制过程中会导致额外的网络、内存、cpu等资源开销，建议在实验停止/结束后再执行日志保存操作。

### <span id="head14"> 三、测试参数设置及推荐环境配置</span>

#### 1、可变设置

1. slave计算节点的GPU卡数：建议将单个物理服务器作为一个slave节点，并使用其所有GPU；
2. 深度学习框架：建议使用keras+tensorflow；
3. 数据集加载方式：建议将数据预处理成TFRecord格式，以加快数据加载的效率；
4. 数据集存储方式：建议采用网络共享存储；
5. 超参搜索：默认从第四轮trial开始，每个trial搜索1次，默认超参为kernel size和batch size。

#### 2、推荐环境配置

- 环境：Ubuntu18.04，docker=19.03.6，SLURM=17.11.2-1build1

- 软件：mindspore-v0.5.1-beta，Ascend910，python3.7.5

- Container：192 cores，755 GB memory,  8NPUs


***NOTE: 推荐基于Kunpeng920 Arm v8-A(192 cores) and Ascend910配置***

## <span id="head17"> Benchmark报告反馈</span>

若测试中遇到问题，请联系renzhx@pcl.ac.cn，并附上`/userhome/mountdir/nni/experiments/experiment_ID/results/`中的html版报告。



## <span id="head18"> 许可</span>

基于 MIT license

感谢“北京技德系统技术有限公司”的协助开发，用于在Atlas800(Ascend910+MindSpore容器环境)上运行该测试工具。
