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
if [ ! -f "node_exporter-1.0.1.linux-amd64.tar.gz" ]; then
    wget  https://github.com/prometheus/node_exporter/releases/download/v1.0.1/node_exporter-1.0.1.linux-amd64.tar.gz
fi


if [ x$install_path != 'x' ];then
    cd $install_path
    #安装node_exporter
    tar -xvf $pkg_path/node_exporter-1.0.1.linux-amd64.tar.gz -C $install_path
    cd node_exporter-1.0.1.linux-amd64
    ./node_exporter 
else
    help
    echo 'd参数错误！请查看帮助信息'
    exit 1
fi
