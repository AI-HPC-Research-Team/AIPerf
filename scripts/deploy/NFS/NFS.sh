#!/bin/bash

#获取本机ip
read -p "请输入本机ip:" GLOBAL_IP

# 所有节点ssh信息
username='root'
password='123123'
port='22'
timeout=10

# 启动全局NFS服务
service nfs-kernel-server start

#创建日志目录
mkdir -p /userhome/NFS_log

# 测试NFS共享文件
touch /userhome/test
touch /localdata/test

# 本机架挂载NFS(7台)
NODE_IP=$GLOBAL_IP
FRAME='第一机架'
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
        sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$NODE_IP "mount $GLOBAL_IP:/localdata /localdata"
	sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$NODE_IP "ls /localdata/test"
        if [ $? -ne 0 ];then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] $NODE_IP 挂载局部NFS错误！" >> /userhome/NFS_log/$FRAME.log
                exit 1
        else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] $NODE_IP 挂载局部NFS成功" >> /userhome/NFS_log/$FRAME.log
        fi
done

# 启动其他机架挂载脚本
cat ip_NFS.txt| while read info;
do
	LOCAL_IP=$(echo $info|awk '{print $1}')
	FRAME=$(echo $info|awk '{print $2}')
	# 传子脚本
	sshpass -p "$password" scp -P $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout sub_NFS.sh $username@$LOCAL_IP:~/
	# 后台执行子脚本，挂载全局NFS和局部NFS
	sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$LOCAL_IP "bash  ~/sub_NFS.sh $GLOBAL_IP $LOCAL_IP $FRAME"
done	
