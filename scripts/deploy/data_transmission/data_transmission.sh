#!/bin/bash

#获取本机ip
read -p "请输入本机ip:" LOCAL_IP

# 数据集路径
IMAGENET_tar='imagenet_tar'

# 所有节点ssh信息
username='root'
password='123123'
port='22'
timeout=10

# 为每个机架第一台机器传输数据集
count=1
cat ip_data_transmission.txt | while read info;
do
	IP=$(echo $info|awk '{print $1}')
	echo $IP
	echo $count
	FRAME=$(echo $info|awk '{print $2}')
	# 传数据集
	sshpass -p "$password" scp -P $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout -r $IMAGENET_tar $username@$IP:~/
	# 传子脚本
	sshpass -p "$password" scp -P $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout sub_data_transmission.sh $username@$IP:~/
	# 后台执行子脚本，机架内做数据集传输
	sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$IP "bash ~/sub_data_transmission.sh $IP $FRAME >/dev/null 2>&1 &"
	count=$((count+1))
done	

# 本机做机架内数据集传输
FRAME='第一机架'
bash sub_data_transmission.sh $LOCAL_IP $FRAME
