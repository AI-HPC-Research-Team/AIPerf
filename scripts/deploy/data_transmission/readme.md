**前提：安装NFS，把本文件夹放在共享目录下 **


#### Instructions
1. pip3 install paramiko
2. Edit 'datapath','codepath'.'username','passward','port','size','folder' in startup.sh (all in string format)
3. Edit remain_ip.txt (represent ips of nodes that with complete dataset)
4. Edit all_ip.txt (represent ips of all nodes, with and without dataset)
5. Run startup.sh
6. Watch running status

#### 说明：(批量传输大文件夹)
0. pip安装paramiko（只要在一个节点安装）
1. 修改startup.sh中的'datapath'和'codepath'（**注：路径结尾不用加斜杠/**）,用户名，密码，文件大小,文件名等参数(注:参数都为字符串形式)
    codepath指当前脚本路径，如'/userhome/data_transmission'
    datapath指数据集路径（指imagenet文件夹所在路径），如datapath='/home/dataset'，则数据集路径结构如下
    ``` 
    /home/dataset (**datapath**)
       |
        --imagenet (**folder**)
           |
            --train.tar
            --val.tar
   ```

2. 修改remain_ip.txt,把**待拷贝**的**没有**数据集的节点ip写入remain_ip.txt (**注：结尾不要有换行符**)
3. 修改all_ip.txt，把**所有**节点ip写入all_ip.txt，包含**有和没有**数据集节点ip
4. 运行startup.sh (在安装了paramiko数据集的某个节点运行)
5. ps -ef查看进程，查看生成的copy.log, 查看是否有errorip.txt, ansible/srun等工具查看传输状态



**注：**
1.若文件大小为G以上，只需输数字，例如145G则size='145'；否则为完整大小，如文件大小为400M，则size='400M'
2.运行之前通过nproc查看cpu核数，all_ip.txt中ip数不能超过核数
3.开启startup.sh之前查看并kill各个节点的进程
```
ps -ef|grep data|grep -v grep|awk '{print $2}'|xargs kill -9
ps -ef|grep rsync|grep -v grep|awk '{print $2}'|xargs kill -9
```

