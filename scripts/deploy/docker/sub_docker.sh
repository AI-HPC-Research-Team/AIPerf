#!/bin/bash
IP=$1
FRAME=$2

# 镜像路径
DOCKET_IMAGE='/root/images'

# 创建日志目录
mkdir -p /userhome/docker_log

# 校验镜像
cd $DOCKET_IMAGE && md5sum -c MD5SUM
if [ $? -ne 0 ];then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 校验码错误！" >> /userhome/docker_log/$FRAME.log
	exit 1
else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 校验成功" >> /userhome/docker_log/$FRAME.log
fi


# 本地导入镜像
docker load -i $DOCKET_IMAGE/*.tar
if [ $? -ne 0 ];then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 镜像导入错误！" >> /userhome/docker_log/$FRAME.log
	exit 1
else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 镜像导入成功" >> /userhome/docker_log/$FRAME.log
fi
# 本地启动容器
启动命令目前未知
if [ $? -ne 0 ];then 
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 容器创建错误！" >> /userhome/docker_log/$FRAME.log
	exit 1
else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 容器创建成功" >> /userhome/docker_log/$FRAME.log
fi

# 节点信息
username='root'
password='123123'
port='22'
timeout=10


# 将镜像传输到机架内其他节点(7台)
for i in {1..7};
do
	# 生成机架内其他ip
	IP=$(echo $IP|awk -F '.' '{print $1"."$2"."$3"."$4+1}')
	sshpass -p "$password" scp -P $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout -r $DOCKET_IMAGE $username@$IP:~/
	# 校验哈希值
	sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$IP "cd $DOCKET_IMAGE && md5sum -c MD5SUM"
        if [ $? -ne 0 ];then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 校验码错误！" >> /userhome/docker_log/$FRAME.log
		exit 1
        else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 校验成功" >> /userhome/docker_log/$FRAME.log
	fi
		
	# 子节点导入镜像
	sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$IP "docker load -i $DOCKET_IMAGE/*.tar"
	if [ $? -ne 0 ];then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 镜像导入错误！" >> /userhome/docker_log/$FRAME.log
		exit 1
        else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 镜像导入成功" >> /userhome/docker_log/$FRAME.log
        fi
	# 子节点启动容器
	sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$IP "启动命令目前未知"
	if [ $? -ne 0 ];then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 容器创建错误！" >> /userhome/docker_log/$FRAME.log
		exit 1
        else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 容器创建成功" >> /userhome/docker_log/$FRAME.log
        fi
done
