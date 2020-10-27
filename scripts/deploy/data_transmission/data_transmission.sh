#!/bin/bash
# ssh information
username=$3
password=$4
port=$5
timeout=100
size=$6 #Gsize of dataset
# absolute file path
IMAGENET_SRC=$2
FOLDER=$7
JPG_PATH=$2/unzip
IMAGENET_TAR=$IMAGENET_SRC/$FOLDER
IMAGENET_DST=$IMAGENET_SRC
ERRORNAME=$1/errorip.txt
IP_FILE=$1/remain_ip.txt

if [ ! -d $IMAGENET_DST ]; then
	mkdir -p $IMAGENET_DST
fi

if [ ! -d $JPG_PATH ]; then
	mkdir -p $JPG_PATH
fi

# loop1 : judge the exsitance of flag file
# remove tmp file
if [ -d $IMAGENET_TAR ]; then
    tmp=$(du -sh $IMAGENET_TAR | awk '{print $1}'|cut -d 'G' -f1)
    if [ ! $tmp == $size ]; then
        rm -rf $IMAGENET_TAR
    fi
fi

while true
do
	if [ -d $IMAGENET_TAR ]; then
        tmp=$(du -sh $IMAGENET_TAR | awk '{print $1}'|cut -d 'G' -f1)
        if [ $tmp == $size ]; then
            # remove empty lines
            sleep $[$RANDOM%30]
            sed -i -e '/^$/d' $IP_FILE
            break
        fi
	fi
done

# loop2 : judge if ips.txt is empty
while true
do
	# if file is locked, continue
	# lock the ip file and sleep
	# read ip
	sleep $[$RANDOM%30]
    if [ ! -s $IP_FILE ]; then
        break
    else
        echo $(cat $IP_FILE |wc -l)
    fi
	IP_TMP=$(head -n +1 $IP_FILE)
    IP_TMP=$(echo $IP_TMP|awk -F '.' '{print $1"."$2"."$3"."$4}')
	echo $IP_TMP
	# remove ip
	sed -i '1d' $IP_FILE
	# unlock ip.txt
	# transfer imagenet
	sshpass -p "$password" rsync -r -e "ssh -p $port -o StrictHostKeyChecking=no -o ConnectTimeout=$timeout" --progress $IMAGENET_TAR $username@$IP_TMP:$IMAGENET_DST
	if [ $? -ne 0 ]; then
		echo $IP_TMP >> $ERRORNAME
	fi
	# sleep random seconds
	sleep $[$RANDOM%30]
done

echo "Data transmission Finished."
echo "Unzipping Local Data."
# unzip train data
#mkdir -p $JPG_PATH/val && mkdir -p $JPG_PATH/train
#cd $IMAGENET_TAR && tar -xf val.tar -C $JPG_PATH/val > /dev/null 2>&1 
#cd $IMAGENET_TAR && tar -xf train.tar -C $JPG_PATH/train >/dev/null 2>&1
echo "Done."
