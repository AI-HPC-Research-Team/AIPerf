#!/bin/bash
username='root'
passward='Pcl@123@'
port='22'
size='9.4'  # GByte or 116M et al
codepath='/userhome/AIPerf/scripts/deploy/data_transmission/v1022'
datapath='/home/dataset'
folder='test1'
# /dataset/
#  --imagenet/
#    --train.tar
#    --val.tar
cd $codepath
python3 sshRemoteCmd.py -code $codepath -data $datapath -uname $username -port $port -key $passward -size $size -folder $folder>remote.log 2>&1
