#!/usr/bin/python3
#coding=utf-8
import argparse
import os
import paramiko
import multiprocessing
paramiko.util.log_to_file("filename.log")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip',type = str,default='/all_ip.txt', help = 'file of ips')
    parser.add_argument('-port',type = str,default='22', help = 'port')
    parser.add_argument('-uname',type = str,default='root', help = 'username')
    parser.add_argument('-key',type = str,default='123123', help = 'password')
    parser.add_argument('-code',type = str,default='/userhome/data_transmission', help = 'code path')
    parser.add_argument('-data',type = str,default='/dataset', help = 'data path')
    parser.add_argument('-size',type = str,default='145', help = 'GBytes of data')
    parser.add_argument('-folder',type = str,default='imagenet', help = 'data folder')
    return parser.parse_args()

def read_file(path):
    if not os.path.exists(path):
        print('IP file does not exist.')
        exit(1)
    
    data = []
    with open(path, "r") as f:
        while True:
            lines = f.readline()
            if not lines.strip():
                break
            data.append(lines.strip())
        
    return data

def ssh_cmd(node, cmd, port, uname, key):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname = str(node), port = int(port),username = str(uname), password = str(key))
    stdin, stdout, stderr = ssh.exec_command(str(cmd))
    ssh.close()

if __name__ == "__main__":
    args = get_args()
    data = read_file(args.code+args.ip)
    max_pool = len(data)
    pool = multiprocessing.Pool(max_pool)
    print('Running......')
    cmd = 'cd ' + args.code + '&&bash data_transmission.sh "'+ args.code + '" "' + args.data + '" "' +args.uname + '" "' + args.key + '" "' +args.port + '" "' + args.size + '" "' + args.folder + '">>copy.log 2>&1'
    print(cmd)
    for node in data:
        print(node)
        pool.apply_async(ssh_cmd,(node, cmd, args.port, args.uname, args.key))
    pool.close()
    pool.join()
    print('Finished.')
