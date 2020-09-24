### 资源监控程序（可选）
资源监控程序使用prometheus收集硬件资源信息，通过grafana将各节点收集到的信息图形化显示在web UI上，有助于在测试过程中对硬件资源使用的实时监控、发现性能瓶颈。
#### master节点分发slave监控插件运行
注意： 所有监控依赖的安装包下载至路径脚本当前路径的.monitortmp文件夹下，可以直接在浏览器下载好之后拷贝到当前路径，提供百度云下载地址：链接：https://pan.baidu.com/s/186bIuqaguoT9j31q-s10wg 
提取码：94be。

```
cd  AIPerf/scripts/monitor
srun -N  节点数  bash monitor_slave_run_nodeexporter.sh -i 安装路径 &
srun -N  节点数  bash monitor_slave_run_dcgmexporter.sh -i 安装路径 &
```
#### master节点执行安装脚本

1）在master节点执行脚本monitor_master_run.sh

```
cd  AIPerf/scripts/monitor
bash monitor_master_run.sh -i 安装路径  -ip slaveip.txt的决定路径
```
#### 访问grafana查看资源信息
打开浏览器访问 master_ip:3000,初始账号密码为admin/admin;

#### grafana增加数据源
在左侧菜单栏按顺序点击以下按钮

configuration ->Data Sources 

点击Prometheus,在URL框中填入master_ip:9090，点击 Save & Test 按钮
#### 导入模板文件
在左侧菜单栏按顺序点击以下按钮

Create -> Import 

点击 Upload .json file 导入 'AIPerf/scripts/monitor/monitor.json'

点击 load 即可看到监控的资源使用情况

#### 其他操作
##### 重启资源监控
当机器重启后监控服务会被关闭，需要手动启动
###### master重启服务
进入到先前安装指定的路径执行
```
/prometheus &
service grafana-server restart
```
###### slave重启服务
```
./node_exporter &
dcgm-exporter &
```
