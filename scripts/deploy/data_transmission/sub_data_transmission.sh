#!/bin/bash
IP=$1
IMAGENET_tar=$2
echo $IP

# 数据集路径
IMAGENET='/root/imagenet'

# 节点信息
username='root'
password='123123'
port='22'
timeout=10

# 本地解压数据集
mkdir -p $IMAGENET/val && mkdir -p $IMAGENET/train
# 解压验证集
cd $IMAGENET_tar && tar -xf val.tar -C $IMAGENET/val > /dev/null 2>&1 &
# 解压训练集
tar -xf $IMAGENET_tar/train.tar -C $IMAGENET/train >/dev/null 2>&1 &

# 将数据集传输到机架内其他节点(7台)
for i in {1..7};
do
	# 生成机架内其他ip
	IP=$(echo $IP|awk -F '.' '{print $1"."$2"."$3"."$4+1}')
	sshpass -p "$password" rsync -r -e "ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout" --progress $IMAGENET_tar $username@$IP:~/
	
	# 创建对应数据集目录
	sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$IP "mkdir -p $IMAGENET/val && mkdir -p $IMAGENET/train" 
	# 解压验证集
	sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$IP "cd $IMAGENET_tar && tar -xf val.tar -C $IMAGENET/val > /dev/null 2>&1 &"
	# 解压训练集
	sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$IP "cd $IMAGENET_tar && tar -xf train.tar -C $IMAGENET/train >/dev/null 2>&1 &"

done
