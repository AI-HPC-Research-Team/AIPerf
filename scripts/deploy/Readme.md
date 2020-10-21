# AIPerf环境部署

本文件涉及3个模块脚本(NFS、docker、dataset)，部署、配置AIPerf正常运行需要的物理环境。

AIPerf系统环境需求：

- 安装sshpass

- 安装openssh-server 并开放root登录权限

- 创建目录/userhome、/localdata，并修改权限为777

- 安装nfs-kernel-server、nfs-common

- 配置文件/etc/exports写入以下内容并重启服务:

  `/userhome   *(rw,sync,insecure,no_root_squash)`
  `/localdata   *(rw,sync,insecure,no_root_squash)`

## 共享文件系统搭建

#### 说明

共享文件系统分为全局NFS和局部NFS，全局NFS以某个机架的第一台机器为服务节点，其他所有节点挂载该节点/userhome；局部NFS以每个机架第一台机器为服务节点，机架内机器挂载该节点/localdata；挂载脚本以全局NFS服务节点为起始节点执行挂载脚本；

#### 运行前注意

1. 以512集群内的某个机架的第一台机器为起点运行NFS.sh，该节点将成为全局NFS服务节点
2. 需要将集群内所有机架的第一台机器的ip以及机架号以空格分隔的形式写入到ip_NFS.txt，如:`192.168.116.12 A10`，运行NFS.sh的机器ip不需要写入ip_NFS.txt
3. 日志文件将会保存在全局共享文件系统/userhome/NFS_log目录下

#### 运行脚本

```
bash NFS.sh
```



##  镜像传输和容器创建

#### 说明

镜像传输以集群的某个机架的第一台机器为起点，根据ip_docker.txt写入的ip顺序将镜像传到每个机架上，当机架接收完镜像后，会在起始机器顺序执行docker.sh的同时机架内调用sub_docker.sh将镜像在本机架内传输，然后导入镜像并创建容器

#### 运行前注意

1. 镜像传输前需要确保全局共享文件系统已经搭建好
2. 以512集群内的某个机架的第一台机器为起点运行docker.sh
3. 运行前需要注意镜像存放在与docker.sh同级的images文件夹内，并通过`md5sum 镜像名 >> MD5SUM`在images内生成校验码文件
4. 需要将集群内所有机架的第一台机器的ip以及机架号以空格分隔的形式写入到ip_docker.txt，如:`192.168.116.12 A10`，运行docker.sh的机器ip不需要写入ip_docker.txt
5. 请确保脚本内的容器启动命令是正确的
6. 日志文件将会保存在全局共享文件系统/userhome/docker_log目录下

#### 运行脚本

```
bash docker.sh
```



## 数据集传输

#### 说明

数据集传输以集群的某个机架的第一台机器为起起始机器根据ip_data_transmission.txt写入的ip顺序将数据集传到每个机架上，当机架接收完数据集后，会在起始机器顺序执行data_transmission.sh的同时机架内调用sub_data_transmission.sh将数据集在本机架内传输。

#### 运行前注意

1. 数据集传输前需要确保全局共享文件系统已经搭建好
2. 以512集群内的某个机架的第一台机器为起点运行data_transmission.sh
3. 运行前需要注意脚本内的数据集路径，该路径下需要包含val.tar和train.tar两个数据集包，并使用`md5sum val.tar train.tar >> MD5SUM`生成校验文件
4. 需要将集群内所有机架的第一台机器的ip以及机架号以空格分隔的形式写入到ip_data_transmission.txt，如:`192.168.116.12 A12`，运行data_transmission.sh的机器ip不需要写入ip_data_transmission.txt
5. 日志文件将会保存在全局共享文件系统/userhome/data_log目录下

#### 运行脚本

```
bash data_transmission.sh
```


