ScripPath=$(pwd)

#删除原有slurm.conf配置文件
rm ${ScripPath}/slurm.conf

#创建slurm.conf文件并写入内容
cat >${ScripPath}/slurm.conf<<EOF
ClusterName=AIPerf
ControlMachine=PCL-DGX2
ControlAddr=192.168.202.110
SlurmctldPort=6817
SlurmdPort=6818
AuthType=auth/munge
StateSaveLocation=/var/spool/slurmd
SlurmdSpoolDir=/var/spool/slurmd
SwitchType=switch/none
MpiDefault=none
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmdPidFile=/var/run/slurmd.pid
ProctrackType=proctrack/pgid
CacheGroups=0
ReturnToService=2
TaskPlugin=task/affinity

# JOB
DefMemPerNode=1024
MaxJobCount=10000
MinJobAge=300

# TIMERS
SlurmctldTimeout=120
SlurmdTimeout=120
InactiveLimit=0
KillWait=30
Waittime=0

# SCHEDULING
SchedulerType=sched/backfill

#SchedulerPort=7321
SelectType=select/cons_res
SelectTypeParameters=CR_CPU_Memory

# LOGGING
SlurmctldDebug=3
SlurmctldLogFile=/var/log/slurmctld.log
SlurmdDebug=3
SlurmdLogFile=/var/log/slurmd.log
JobCompType=jobcomp/none

#JobCompLoc=
JobAcctGatherType=jobacct_gather/none

#NODE INFO
EOF

#所有节点ssh信息
username='root'
password='123123'
port='222'
timeout=10

#获取所有slave机器中最少的cpu核数
min_CPUs=9999
for host in $(cat slaveip.txt);
do
    slave_CPUs=$(sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$host nproc)
    if [ $slave_CPUs -lt $min_CPUs ];then
        min_CPUs=$slave_CPUs
    fi
done

#添加slave机器信息
for host in $(cat slaveip.txt);
do
    sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$host touch /etc/slurm-llnl/slurm.conf
    result=""
    result=$(sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$host slurmd -C)
    echo $(echo $result|awk '{for (f=1;f<=NF;f+=1){if ($f ~ /NodeName/ || $f ~ /RealMemory/){print $f}}{print "CPUs="'$min_CPUs'}}')>>${ScripPath}/slurm.conf
    sed -i "$ s/$/ NodeAddr=$host/" ${ScripPath}/slurm.conf
done
echo "PartitionName=compute Nodes=ALL Default=YES Shared=YES MaxTime=INFINITE State=UP" >> ${ScripPath}/slurm.conf

#替换master机器信息
MasterIP=$(cat masterip.txt)
MasterHostname=$(sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$MasterIP hostname)
sed -i "s/^ControlMachine.*/ControlMachine=$MasterHostname/" ${ScripPath}/slurm.conf
sed -i "s/^ControlAddr.*/ControlAddr=$MasterIP/" ${ScripPath}/slurm.conf

#master启动服务
sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$MasterIP rm /etc/slurm-llnl/slurm.conf
sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$MasterIP ln -s ${ScripPath}/slurm.conf /etc/slurm-llnl/slurm.conf
sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$MasterIP service munge restart
sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$MasterIP service slurmctld restart

#slave启动服务
for host in $(cat slaveip.txt);
do
    sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$host rm /etc/slurm-llnl/slurm.conf
    sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$host ln -s ${ScripPath}/slurm.conf /etc/slurm-llnl/slurm.conf
    sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$host service munge restart
    sshpass -p "$password" ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout $username@$host service slurmd restart
done
echo -e "\033[32mslurm配置所有节点最大可用CPU核数为: $min_CPUs\033[0m"
echo -e "\033[32m请记住此信息并关联后续config.yml配置 'srun --cpus-per-task=$(($min_CPUs-1))'\033[0m"
