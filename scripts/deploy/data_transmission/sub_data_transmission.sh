#!/bin/bash
IP=$1
FRAME=$2


echo $IP
echo $FARME

# 数据集路径
IMAGENET_tar='/root/imagenet_tar'
IMAGENET='/root/imagenet'

# 节点信息
username='root'
password='123123'
port='22'
timeout=10

# 创建日志目录
mkdir -p /userhome/data_log

# 校验本地数据集哈希值
cd $IMAGENET_tar && md5sum -c MD5SUM
if [ $? -ne 0 ];then
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP校验码错误！" >> /userhome/data_log/$FRAME.log
else
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 校验成功" >> /userhome/data_log/$FRAME.log
fi

# 本地解压数据集
mkdir -p $IMAGENET/val && mkdir -p $IMAGENET/train
# 解压验证集
cd $IMAGENET_tar && tar -xf val.tar -C $IMAGENET/val > /dev/null 2>&1 &
# 解压训练集
tar -xf $IMAGENET_tar/train.tar -C $IMAGENET/train && cd $IMAGENET/train && for i in *.tar;do mkdir -p ${i%.tar}; tar -xvf $i -C ${i%.tar};rm -rf $i;done >/dev/null 2>&1 &

# 将数据集传输到机架内其他节点(7台)
for i in {1..7};
do
	# 生成机架内其他ip
	IP=$(echo $IP|awk -F '.' '{print $1"."$2"."$3"."$4+1}')
	sshpass -p "$password" scp -P $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout -r $IMAGENET_tar $username@$IP:~/
	# 校验数据集哈希值
	sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$IP "cd $IMAGENET_tar && md5sum -c MD5SUM"
	if [ $? -ne 0 ];then
		echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 校验码错误！" >> /userhome/data_log/$FRAME.log
	else
		echo "[$(date '+%Y-%m-%d %H:%M:%S')] $IP 校验成功" >> /userhome/data_log/$FRAME.log
		# 创建对应数据集目录
		sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$IP "mkdir -p $IMAGENET/val && mkdir -p $IMAGENET/train" 
		# 解压验证集
		sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$IP "cd $IMAGENET_tar && tar -xf val.tar -C $IMAGENET/val > /dev/null 2>&1 &"
		# 解压训练集
		sshpass -p "$password" ssh -n -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$IP "cd $IMAGENET_tar && tar -xf train.tar -C $IMAGENET/train && cd $IMAGENET/train && for i in *.tar;do mkdir -p \${i%.tar}; tar -xvf \$i -C \${i%.tar};rm -rf \$i;done >/dev/null 2>&1 &"
	fi
done
