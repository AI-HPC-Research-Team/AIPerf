#!/bin/bash
GLOBAL_IP=$1
LOCAL_IP=$2
FRAME=$3

# 挂载全局NFS
mount $GLOBAL_IP:/userhome /userhome
ls /userhome/test
if [ $? -ne 0 ];then
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] $LOCAL_IP 挂载全局NFS错误！(机架第一节点)" >> /userhome/NFS_log/$FRAME.log
        exit 1
else
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] $LOCAL_IP 挂载全局NFS成功(机架第一节点)" >> /userhome/NFS_log/$FRAME.log
fi

# 本机启动局部NFS服务
#service nfs-kernel-server start
#if [ $? -ne 0 ];then
#        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $LOCAL_IP 启动局部NFS错误！" >> /userhome/NFS_log/$FRAME.log
#        exit 1
#else
#        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $LOCAL_IP 启动NFS成功" >> /userhome/NFS_log/$FRAME.log
#fi

# 测试NFS共享文件
touch /localdata/test

# 节点信息
username='root'
password='123123'
port='22'
timeout=10

# 机架内节点挂载全局NFS和机架内局部NFS(7台)
NODE_IP=$LOCAL_IP
for i in {1..7};
do
	# 生成机架内其他ip
	NODE_IP=$(echo $NODE_IP|awk -F '.' '{print $1"."$2"."$3"."$4+1}')
	# 挂载全局NFS
	sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$NODE_IP "mount $GLOBAL_IP:/userhome /userhome"
	sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$NODE_IP "ls /userhome/test"
	if [ $? -ne 0 ];then
		echo "[$(date '+%Y-%m-%d %H:%M:%S')] $NODE_IP 挂载全局NFS错误！" >> /userhome/NFS_log/$FRAME.log
		exit 1
	else
		echo "[$(date '+%Y-%m-%d %H:%M:%S')] $NODE_IP 挂载全局NFS成功" >> /userhome/NFS_log/$FRAME.log
	fi
	# 挂载局部NFS
	sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$NODE_IP "mount $LOCAL_IP:/localdata /localdata"
	sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$NODE_IP "ls /localdata/test"
	if [ $? -ne 0 ];then
		echo "[$(date '+%Y-%m-%d %H:%M:%S')] $NODE_IP 挂载局部NFS错误！" >> /userhome/NFS_log/$FRAME.log
		exit 1
	else
		echo "[$(date '+%Y-%m-%d %H:%M:%S')] $NODE_IP 挂载局部NFS成功" >> /userhome/NFS_log/$FRAME.log
	fi
done
