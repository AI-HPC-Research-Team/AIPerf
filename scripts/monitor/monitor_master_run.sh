#!/bin/bash
function help()
{
        echo '用法：[-i|--install 参数 -p|--packge 参数]'
        echo '  或：[-r|--run]'
        echo '-h|--help    打印帮助信息'
        echo '-i|--install 程序安装路径'
        echo '-ip slaveip.txt路径'

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
        -r|--run)
            run=true
            shift 2
            ;;
        -ip)
            ip_path=$2
            echo $ip_path******************
            shift 2
            ;;
        *)
            help
            echo '参数错误！请查看帮助信息'
            exit 1
    esac
done
sh_path=$PWD
pkg_path=$PWD/.monitortmp
if [ ! -d $pkg_path ]; then
    mkdir $pkg_path 
fi

cd $pkg_path 
if [ ! -f "prometheus-2.19.2.linux-amd64.tar.gz" ]; then
    wget https://github.com/prometheus/prometheus/releases/download/v2.19.2/prometheus-2.19.2.linux-amd64.tar.gz
fi

if [ ! -f "grafana_7.0.4_amd64.deb" ]; then
    wget https://dl.grafana.com/oss/release/grafana_7.0.4_amd64.deb
fi


if [ x$install_path != 'x' -a x$ip_path != 'x' ];then
    #安装Prometheus
    tar -xvf $pkg_path/prometheus-2.19.2.linux-amd64.tar.gz -C $install_path
    cd $install_path/prometheus-2.19.2.linux-amd64
    #安装grafana
    apt install adduser libfontconfig1 -y
    dpkg -i $pkg_path/grafana_7.0.4_amd64.deb

	#替换ip
    iplist=$(cat $ip_path)
    for ip in $iplist
    do 
        if [ x$cpuconcat != 'x' ];then
            cpuconcat=$(echo $cpuconcat,)
        fi
        cpuconcat=$(echo "$cpuconcat\'$ip:9100\'")
        if [ x$gpuconcat != 'x' ];then
            gpuconcat=$(echo $gpuconcat,)
        fi
        gpuconcat=$(echo "$gpuconcat\'$ip:9400\'")
    done
        
    cp $sh_path/prometheus.yml  $sh_path/prometheus.yml.new
    sed -i "s/'localhost:9100'/$cpuconcat/g" $sh_path/prometheus.yml.new
    sed -i "s/'localhost:9400'/$gpuconcat/g" $sh_path/prometheus.yml.new

    #替换Prometheus
    cp $sh_path/prometheus.yml.new $install_path/prometheus-2.19.2.linux-amd64/prometheus.yml
    cd $install_path/prometheus-2.19.2.linux-amd64
    #./prometheus &
    #service grafana-server restart
else
    help
    echo '参数错误！请查看帮助信息'
    exit 1
fi
