# 简介
为实验集群搭建slurm环境，本程序默认所有节点已经完成slurm、munge安装以及munge秘钥配置
# 执行前检查
确保本目录 autoconfig_slurm 在所有节点共享的共享目录下
## 检查sshpass
确保所有节点安装了sshpass

```
sshpass -V
```
如果没有安装则执行

```
sudo apt install sshpass -y
```
## 检查ssh服务
确保所有节点已经安装ssh，并启动服务

```
sudo service ssh status
```
### 如果没有安装则

```
sudo apt install openssh-client openssh-server -y
```
开启ssh登录权限,修改ssh配置文件 /etc/ssh/sshd_config

```
vim /etc/ssh/sshd_config
```
找到PermitRootLogin prohibit-password所在行，并修改为<br>

```
#PermitRootLogin prohibit-password<br>
PermitRootLogin yes
```
重启ssh服务

```
sudo service ssh restart
```
### 如果服务没有开启则

```
sudo service ssh start
```
# 运行程序
1. 将所有slave节点ip按行写入slaveip.txt
2. 将master节点ip写入masterip.txt
3. 确保所有节点的ssh用户、密码、端口是一致的，并根据自身情况修改 slurm_autoconfig.sh脚本中的用户名和密码
在master节点运行
```
bash slurm_autoconfig.sh
```
# 运行后检查
执行命令查看所有节点状态
```
sinfo
```
如果所有节点STATE列为idle则slurm配置正确，运行正常。<br>
如果STATE列为unk，等待一会再执行sinfo查看，如果都为idle，则slurm配置正确，运行正常。<br>