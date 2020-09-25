# 一、系统安装

Ubuntu Arm Server通过IBMC进行系统安装（由华为提供账号），系统安装完成后，可按以下方式进行网络配置：

```
$ip link add lin
$ip link add link enp61s0f0 name enp61s0f0.206 type vlan id 206
$ifconfig enp61s0f0 up
$ifconfig enp61s0f0.206 up
$ifconfig enp61s0f0.206 192.168.206.25/24				#ip根据实际情况设置
$ip r add default via 192.168.206.1 dev enp61s0f0.206
$echo "nameserver 114.114.114.114" > /etc/resolv.conf
$systemctl restart sshd
```


# 二、Ubuntu Arm系统环境配置

[查看PDF](https://support.huaweicloud.com/instg-9000-A800_9000_9010/instg-9000-A800_9000_9010.pdf)

## 前提条件

- 验证Linux操作系统版本信息

执行**uname -m && cat /etc/\*release**命令，查询正在运行的操作系统版本和操作系统架构。

系统正在运行的操作系统版本和操作系统架构必须与下表中要求一致。

| 硬件形态               | host操作系统版本 | 软件包默认的host操作系统内核版本               | GCC编译器版本 | GLIBC版本 |
| ---------------------- | ---------------- | ---------------------------------------------- | ------------- | --------- |
| aarch64+Atlas 800 9000 | EulerOS 2.8      | 4.19.36-vhulk1907.1.0.h453.eulerosv2r8.aarch64 | 7.3.0         | 2.28      |
| aarch64+Atlas 800 9000 | Ubuntu 18.04.2   | 4.15.0-45-generic                              | 7.4.0         | 2.27      |
| aarch64+Atlas 800 9000 | CentOS 7.6       | 4.14.0-115.el7a.0.1.aarch64                    | 4.8.5         | 2.17      |


- 验证Linux操作系统内核版本
  执行**uname -r**命令，查询当前host操作系统的内核版本。

## 检查root用户的umask

1. 以root用户登录安装环境。
2. 检查root用户的umask值。
```
$umask
```

3. 如果umask不等于0022，请执行如下操作配置，在该文件的最后一行添加umask 0022后保存。
```
$vi ~/.bashrc 
$source ~/.bashrc
```
## 创建安装及运行用户
昇腾芯片驱动和固件、CANN软件使用的安装和运行用户如下表所示。

| 软件包类型               | 安装用户                                     | 运行用户                                                     |
| ------------------------ | -------------------------------------------- | ------------------------------------------------------------ |
| 昇腾芯片驱动和固件安装包 | 必须为root。                                 | 必须为**HwHiAiUser**，且**HwHiAiUser**需在昇腾芯片驱动和固件安装前创建。 |
| CANN软件包               | root或非root均可，具体参见表格下方后续内容。 | 必须非root，具体参见表格下方后续内容。                       |

使用root用户或非root用户安装CANN软件前都必须创建HwHiAiUser用户。

1. 创建HwHiAiUser用户。
切换到root用户下，执行如下命令创建HwHiAiUser用户。
```
$groupadd HwHiAiUser           //创建HwHiAiUser用户属组
$useradd -g HwHiAiUser -d /home/HwHiAiUser -m HwHiAiUser            //创建HwHiAiUser用户，其属组为HwHiAiUser
```
其中HwHiAiUser用户属组必须指定为HwHiAiUser。

2. 安装方式。
- 若使用root用户安装
可以使用root用户进行安装，但必须创建HwHiAiUser用户，安装完CANN软件之后要求使用非root用户运行（运行用户只能为HwHiAiUser用户属组）。

- 若使用非root用户安装
该场景下安装及运行用户必须相同。

- 若已有非root用户，必须将此用户加入HwHiAiUser用户属组，然后切换到此用户正常安装即可，默认的安装用户及运行用户为当前用户。
- 若想使用新的非root用户，则需要先创建该用户，请参见如下方法创建。

创建非root用户操作方法如下，如下命令请以root用户执行。

1. 创建非root用户。
```
$useradd -g HwHiAiUser -d /home/username -m username
```
*username*由用户自定义，但其属组必须指定为HwHiAiUser。

2. 设置非root用户密码。
```
$passwd username
```

说明：

- 创建的运行用户不能为root用户属组。
- 创建完HwHiAiUser用户后， 请勿关闭该用户的登录认证功能。
- 密码有效期为90天，您可以在/etc/login.defs文件中修改有效期的天数。

## 修改源
深度学习引擎包、实用工具包和框架插件包安装过程需要下载相关依赖，请确保安装环境能够连接网络。
修改/etc/apt/sources.list 文件为如下内容：

```
deb http://mirrors.aliyun.com/ubuntu-ports/ bionic main restricted
deb http://mirrors.aliyun.com/ubuntu-ports/ bionic-updates main restricted
deb http://mirrors.aliyun.com/ubuntu-ports/ bionic universe
deb http://mirrors.aliyun.com/ubuntu-ports/ bionic-updates universe
deb http://mirrors.aliyun.com/ubuntu-ports/ bionic multiverse
deb http://mirrors.aliyun.com/ubuntu-ports/ bionic-updates multiverse
deb http://mirrors.aliyun.com/ubuntu-ports/ bionic-backports main restricted universe multiverse
deb http://mirrors.aliyun.com/ubuntu-ports bionic-security main restricted
deb http://mirrors.aliyun.com/ubuntu-ports bionic-security universe
deb http://mirrors.aliyun.com/ubuntu-ports bionic-security multiverse
```

请在root用户下执行如下命令检查源是否可用。
```
$apt-get update
```
如果命令执行报错或者后续安装依赖时等待时间过长甚至报错，则检查网络是否连接或者把“/etc/apt/sources.list”文件中的源更换为可用的源或使用镜像源。

## 配置安装用户权限

当用户使用非root用户安装时，需要操作该章节；当用户使用root用户安装时，可忽略该章节。

深度学习引擎包、实用工具包和框架插件包安装前需要下载相关依赖软件，下载依赖软件需要使用**sudo apt-get**权限，请以root用户执行如下操作。

1. 安装sudo，使用如下命令安装。

```
$apt-get install sudo
```
2. 打开“/etc/sudoers”文件：
```
$chmod u+w /etc/sudoers
$vi /etc/sudoers
```
3. 在该文件“# User privilege specification”下面增加如下内容：
```
username ALL=(ALL:ALL)   NOPASSWD:SETENV:/usr/bin/apt-get, /usr/bin/unzip, /usr/bin/pip, /bin/tar, /bin/mkdir, /bin/rm, /bin/sh, /bin/cp, /bin/bash, /usr/bin/make install, /bin/ln -s /usr/local/python3.7.5/bin/python3 /usr/bin/python3.7, /bin/ln -s /usr/local/python3.7.5/bin/pip3 /usr/bin/pip3.7, /bin/ln -s /usr/local/python3.7.5/bin/python3 /usr/bin/python3.7.5, /bin/ln -s /usr/local/python3.7.5/bin/pip3 /usr/bin/pip3.7.5
```
**“username”**为执行安装脚本的普通用户名。
说明：
请确保“/etc/sudoers”文件的最后一行为**“#includedir /etc/sudoers.d”**，如果没有该信息，请手动添加。

4. 添加完成后，执行**:wq!**保存文件。
5. 执行以下命令取消“/etc/sudoers”文件的写权限：
```
$chmod u-w /etc/sudoers
```

## 环境要求
- 如果使用root用户安装python及其依赖请执行步骤1至3，注意，步骤1至3命令中有sudo的需要删除sudo。
- 如果python及其依赖是使用非root用户安装，则需要使用**su - \*username\***命令切换到非root用户继续执行步骤1至3。

### 步骤1

检查系统是否安装python依赖以及gcc等软件。
分别使用如下命令检查是否安装gcc，make以及python依赖软件等。

```
gcc --version
make --version
cmake --version
dpkg -l zlib1g| grep zlib1g| grep ii
dpkg -l zlib1g-dev| grep zlib1g-dev| grep ii
dpkg -l libbz2-dev| grep libbz2-dev| grep ii
dpkg -l libsqlite3-dev| grep libsqlite3-dev| grep ii
dpkg -l openssl| grep openssl| grep ii
dpkg -l libssl-dev| grep libssl-dev| grep ii
dpkg -l libxslt1-dev| grep libxslt1-dev| grep ii
dpkg -l libffi-dev| grep libffi-dev| grep ii
dpkg -l unzip| grep unzip| grep ii
dpkg -l pciutils| grep pciutils| grep ii
dpkg -l net-tools| grep net-tools| grep ii
dpkg -l libblas-dev| grep libblas-dev| grep ii
dpkg -l gfortran| grep gfortran| grep ii
dpkg -l libblas3| grep libblas3| grep ii
dpkg -l libopenblas-dev| grep libopenblas-dev| grep ii
```
若分别返回如下信息则说明已经安装，进入下一步。

```
gcc (Ubuntu/Linaro 7.5.0-3ubuntu1~18.04) 7.5.0
GNU Make 4.1
cmake version 3.10.2
zlib1g:arm64   1:1.2.11.dfsg-0ubuntu2 arm64        compression library - runtime
zlib1g-dev:arm64 1:1.2.11.dfsg-0ubuntu2 arm64        compression library - development
libbz2-dev:arm64 1.0.6-8.1ubuntu0.2 arm64        high-quality block-sorting file compressor library - development
libsqlite3-dev:arm64 3.22.0-1ubuntu0.3 arm64        SQLite 3 development files
openssl  1.1.1-1ubuntu2.1~18.04.6 arm64   Secure Sockets Layer toolkit - cryptographic utility
libssl-dev:arm64 1.1.1-1ubuntu2.1~18.04.6 arm64     Secure Sockets Layer toolkit - development files
libxslt1-dev:arm64 1.1.29-5ubuntu0.2 arm64        XSLT 1.0 processing library - development kit
libffi-dev:arm64 3.2.1-8      arm64        Foreign Function Interface library (development files)
unzip          6.0-21ubuntu1 arm64        De-archiver for .zip files
pciutils       1:3.5.2-1ubuntu1 arm64        Linux PCI Utilities
net-tools      1.60+git20161116.90da8a0-1ubuntu1 arm64        NET-3 networking toolkit
libblas-dev:arm64 3.7.1-4ubuntu1 arm64        Basic Linear Algebra Subroutines 3, static library
gfortran       4:7.4.0-1ubuntu2.3 arm64        GNU Fortran 95 compiler
libblas3:arm64 3.7.1-4ubuntu1 arm64     Basic Linear Algebra Reference implementations, shared library
libopenblas-dev:arm64 0.2.20+ds-4  arm64    Optimized BLAS (linear algebra) library (development files)
```
否则请执行如下安装命令（如果只有部分软件未安装，则如下命令修改为只安装还未安装的软件即可）：

```
$sudo apt-get install -y gcc make cmake zlib1g zlib1g-dev libbz2-dev openssl libsqlite3-dev libssl-dev libxslt1-dev libffi-dev unzip pciutils net-tools libblas-dev gfortran libblas3 libopenblas-dev docker.io libglib2.0-dev
```
### 步骤2

检查系统是否安装python开发环境。
开发套件包依赖python环境，分别使用命令**python3.7.5 --version**、**pip3.7.5 --version**检查是否已经安装，如果返回如下信息则说明已经安装，进入下一步。
```
Python 3.7.5
pip 19.2.3 from /usr/local/python3.7.5/lib/python3.7/site-packages/pip (python 3.7)
```
否则请根据如下方式安装python3.7.5。

1. 使用wget下载python3.7.5源码包，可以下载到安装环境的任意目录，命令为：

```
$wget https://www.python.org/ftp/python/3.7.5/Python-3.7.5.tgz
```
2. 进入下载后的目录，解压源码包，命令为：

```
$tar -zxvf Python-3.7.5.tgz
```
3. 进入解压后的文件夹，执行配置、编译和安装命令：

```
$cd Python-3.7.5
$./configure --prefix=/usr/local/python3.7.5 --enable-shared
$make
$sudo make install
```
其中**“--prefix”**参数用于指定python安装路径，用户根据实际情况进行修改。**“--enable-shared”**参数用于编译出libpython3.7m.so.1.0动态库。

本手册以--prefix=/usr/local/python3.7.5路径为例进行说明。执行配置、编译和安装命令后，安装包在/usr/local/python3.7.5路径，libpython3.7m.so.1.0动态库在/usr/local/python3.7.5/lib/libpython3.7m.so.1.0路径。

4. 查询/usr/lib64或/usr/lib下是否有libpython3.7m.so.1.0，若有则跳过此步骤或将系统自带的libpython3.7m.so.1.0文件备份后执行如下命令：
```
$sudo cp /usr/local/python3.7.5/lib/libpython3.7m.so.1.0 /usr/lib64
```
如果出现如下显示，则输入y，表示覆盖系统自带的libpython3.7m.so.1.0文件。
cp: overwrite 'libpython3.7m.so.1.0'?

如果环境上没有/usr/lib64，则复制到/usr/lib目录：

```
$sudo cp /usr/local/python3.7.5/lib/libpython3.7m.so.1.0 /usr/lib
```
libpython3.7m.so.1.0文件所在路径请根据实际情况进行替换。

5. 执行如下命令设置软链接：

```
$sudo ln -s /usr/local/python3.7.5/bin/python3 /usr/bin/python3.7
$sudo ln -s /usr/local/python3.7.5/bin/pip3 /usr/bin/pip3.7
$sudo ln -s /usr/local/python3.7.5/bin/python3 /usr/bin/python3.7.5
$sudo ln -s /usr/local/python3.7.5/bin/pip3 /usr/bin/pip3.7.5
```
执行上述软链接时如果提示链接已经存在，则可以先执行如下命令删除原有链接然后重新执行。
```
$sudo rm -rf  /usr/bin/python3.7.5
$sudo rm -rf  /usr/bin/pip3.7.5
$sudo rm -rf  /usr/bin/python3.7
$sudo rm -rf  /usr/bin/pip3.7
```
6. 安装完成之后，执行如下命令查看安装版本，如果返回相关版本信息，则说明安装成功。
```
$python3.7.5 --version
$pip3.7.5  --version
```
### 步骤3

需要安装的依赖列表如下表所示。

| 依赖名称     | 版本号   | 安装命令                         |
| ------------ | -------- | -------------------------------- |
| Python       | 3.7.5    | 检查系统是否安装python开发环境。 |
| numpy        | >=1.13.3 | pip3.7 install numpy             |
| decorator    | >=4.4.0  | pip3.7 install decorator         |
| sympy        | 1.4      | pip3.7 install sympy==1.4        |
| cffi         | 1.12.3   | pip3.7 install cffi              |
| pyyaml       | -        | pip3.7 install pyyaml            |
| pathlib2     | -        | pip3.7 install pathlib2          |
| grpcio       | -        | pip3.7 install grpcio            |
| grpcio-tools | -        | pip3.7 install grpcio-tools      |
| protobuf     | -        | pip3.7 install protobuf          |
| scipy        | -        | pip3.7 install Scipy             |
| requests     | -        | pip3.7 install requests          |

请安装前完成相关依赖的安装。如下命令如果使用非root用户安装，需要在安装命令后加上**--user**，例如：

# 三、安装驱动和固件

请按照“驱动 > 固件”的顺序，分别安装组件软件包。每个组件软件包的安装步骤相同，详细操作如下。软件包中的*****请根据实际情况进行替换。

1. 使用**root**用户登录到运行环境，将*.run软件包上传至到运行环境任意路径下，如/opt下。
2. 增加安装用户对软件包的可执行权限。

在软件包所在路径执行**ls -l**命令检查安装用户是否有该文件的执行权限，若没有，请执行如下命令。
```
chmod +x \*.run
```

3. 执行如下命令，校验软件包安装文件的一致性和完整性。
```
./\*.run --check
```

4. （可选）设置安装路径。

- 如果用户需要指定软件包安装路径，则该步骤必选。
1. 创建安装路径：**mkdir /test/HiAI/**
2. 为创建的路径加权限：**chmod 550 /test/HiAI/**
3. 为创建的路径设置属主：**chown HwHiAiUser:HwHiAiUser** **/test/HiAI/**
- 如果用户不需要指定安装路径，请忽略该步骤，安装包会安装到默认路径“**/usr/local/Ascend**”下。

5. 执行以下命令进行安装。
- 如果软件包安装路径是指定的，执行**./\*.run** **--run** **--install-path=/test/HiAI/**
- 如果软件包安装路径是默认的，执行./*.run --run，如下：
```
$./A800-9000-NPU_Driver-20.0.RC1-ARM64-Ubuntu18.04.run
$./A800-9000-NPU_Firmware-1.73.1005.1.b050.run
$./Ascend-Toolbox-20.0.RC1-arm64-linux_gcc7.3.0.run
```

说明：
- 软件包默认安装路径：/usr/local/Ascend
- 安装详细日志路径：/var/log/ascend_seclog/ascend_install.log
- 安装后软件包的安装路径、安装命令以及运行用户信息记录路径：Driver/Firmware：/etc/ascend_install.info

6. 若出现如下信息，则说明安装成功：
- 驱动： 
```
Driver package install success! Reboot needed for installation/upgrade to take effect!
```
- 固件：
```
Firmware package install success! Reboot needed for installation/upgrade to take effect!
```

7. 重启运行环境。
8. 查看安装的驱动版本号。
在软件包的安装路径下，例如root用户默认路径“/usr/local/Ascend/${package_name}”，执行如下命令查看所升级软件包版本是否正确。
```
$cat version.info
Version=1.73.T105.0.B050
```

9. 查看安装的NPU固件版本号。
```
$/usr/local/Ascend/driver/tools/upgrade-tool --device_index -1 --component -1 --version
```

```
Get component version(1.73.5.0.b050) succeed for deviceId(0), componentType(0). 
{"device_id":0, "component":nve, "version":1.73.5.0.b050} 
Get component version(1.73.5.0.b050) succeed for deviceId(0), componentType(3). 
{"device_id":0, "component":uefi, "version":1.73.5.0.b050} 
Get component version(1.73.5.0.b050) succeed for deviceId(0), componentType(8). 
{"device_id":0, "component":imu, "version":1.73.5.0.b050} 
Get component version(1.73.105.0.b050) succeed for deviceId(0), componentType(9). 
{"device_id":0, "component":imp, "version":1.73.105.0.b050}
```

10. 执行**npu-smi info**查看NPU工具安装是否成功。
回显如下类似信息，说明安装成功。否则，说明安装失败，请联系华为技术支持处理。
![img](https://download.huawei.com/mdl/imgDownload?uuid=d5709732df57416ca0d74aa1e987adb4)


# 四、版本安装注意事项

- 使用run安装包后，不要手动设置环境变量export LD_LIBRARY_PATH指向之前rar包的旧SO文件，否则可能会出现run安装包内工具连接到之前版本的动态库。指向第三方库文件路径、非run安装包发布库文件路径的配置不受影响。

- 查看日志时需注意：日志时间采用的是系统时间，device侧时间与host侧时间保持同步，修改host侧时间可以使用"date"命令。

例如：执行**date -s 17:55:55**命令，将系统时间设定为17点55分55秒。
