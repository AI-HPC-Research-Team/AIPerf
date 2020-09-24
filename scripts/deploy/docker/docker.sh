#!/bin/bash

#获取本机ip
read -p "请输入本机ip:" LOCAL_IP

# 镜像路径
DOCKER_IMAGE='images'

# 所有节点ssh信息
username='root'
password='123123'
port='22'
timeout=10

# 为每个机架第一台机器传输镜像
count=1
cat ip_docker.txt | while read info;
do
	IP=$(echo $info|awk '{print $1}')
	echo $IP
        echo $count
        FRAME=$(echo $info|awk '{print $2}')
	# 传数镜像
	sshpass -p "$password" scp -P $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout -r $DOCKER_IMAGE $username@$IP:/root
	# 传子脚本
	sshpass -p "$password" scp -P $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout sub_docker.sh $username@$IP:/root
	# 后台执行子脚本，机架内做镜像传输
	sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$IP "bash ~/sub_docker.sh $IP $FRAME >/dev/null 2>&1 &"
	count=$((count+1))
done	

# 本机做机架内镜像
FRAME='第一机架'
bash sub_docker.sh $LOCAL_IP
