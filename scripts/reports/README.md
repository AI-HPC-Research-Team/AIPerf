### Run the following script to generate final report:

当测试运行后，通过report程序会在终端打印experiment的Error、Score、Regulated Score等信息.
#### 用法
python3 report.py --id experiment_id --logs True
##### 参数说明
--id 必须参数，运行测试的试验号

--logs 可选参数,参数为 True或 False(默认)

##### 结果输出
当只有--id时,在终端打印experiment的Error、Score、Regulated Score等信息

同时会产生实验报告存放在experiment_ID的对应路径/root/mountdir/nni/experiments/experiment_ID/results目录下

实验成功时报告为 Report_Succeed.html

实验失败时报告为 Report_Failed.html

实验失败会报告失败原因，请查阅AI Benchmark测试规范分析失败原因

当--logs为True时，会将日志和数据拷贝到同一目录/root/mountdir/nni/experiments/experiment_ID/results/logs下，由于实验数据在复制过程中会导致额外的网络、内存、cpu、等资源开销，建议在实验结束/停止后再执行日志保存操作。

