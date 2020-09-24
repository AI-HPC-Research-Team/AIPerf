#!/bin/bash
function help()
{
        echo '用法：[-i|--install 参数 -p|--packge 参数]'
        echo '-h|--help    打印帮助信息'
        echo '-i|--install 程序安装路径'
}
while [ $# -gt 0 ]; do
    case $1 in
        -h|--help)
            help
            exit 0 
            ;;
        -i|--install)
            install_path=$2
            shift 2
            ;;
        *)
            help
            echo '参数错误！请查看帮助信息'
            exit 1
    esac
done

pkg_path=$PWD/.monitortmp
if [ ! -d $pkg_path ]; then
    mkdir $pkg_path 
fi

cd $pkg_path 

if [ ! -f "datacenter-gpu-manager_1.7.2_amd64.deb" ]; then
    wget http://note.youdao.com/yws/public/resource/913f0fb8eb1972053173f7b77c7fc803/xmlnote/WEBRESOURCE73800b397ab75ec07fa3d37765b0de1e/3661
    mv 3661 datacenter-gpu-manager_1.7.2_amd64.deb
fi

if [ ! -f "go1.14.6.linux-amd64.tar.gz" ]; then
    wget https://golang.google.cn/dl/go1.14.6.linux-amd64.tar.gz
fi

if [ ! -f "gpu-monitoring-tools" ]; then                                                                                                                                                
    git clone https://github.com/NVIDIA/gpu-monitoring-tools.git-                                                                                                                   
fi 


if [ x$install_path != 'x' ];then
    dpkg -i $pkg_path/datacenter-gpu-manager_1.7.2_amd64.deb
    tar -C /usr/local/ -xzf $pkg_path/go1.14.6.linux-amd64.tar.gz
    PATH=$PATH:/usr/local/go/bin
    cd $pkg_path/gpu-monitoring-tools
    make binary
    make install
    dcgm-exporter 
else
    help
    echo 'd参数错误！请查看帮助信息'
    exit 1
fi
