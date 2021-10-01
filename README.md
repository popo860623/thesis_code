# Environment Setup
## Introduction
Deploy thesis code on ubuntu 18.04

## Requirement
* 安裝Git，將project clone下來
### Git
```
sudo apt-get update
sudo  apt install git
```
### Git clone project
```
https://github.com/popo860623/thesis_code.git
```

### requirements
* 安裝建立環境所使用的套件，Mininet則透過Github clone安裝
```
sudo apt install python3-ryu
sudo apt install python3-pip
sudo apt install python-pip
sudo pip3 install networkx=2.5.1
sudo pip install networkx
sudo pip3 install tensorflow==1.14
sudo pip3 install numpy==1.19.5
sudo pip install numpy==1.16.6
sudo pip3 install pandas=1.1.5
```
### Mininet installation
[mininet install](https://hackmd.io/3N2L8HzzQhKqw6hXQDZhcg?view)

## Quick start
### Start mininet
```=shell
$ cd ~/thesis_code/Topology_and_data_generation/mininet/
$ sudo python nsfnet_topo.py
```

https://user-images.githubusercontent.com/53852453/134792984-486a5732-6471-4a42-9d1d-ebb970e3b0b2.mp4


### Start controller
```shell=
$ cd /home/hao/thesis_code/controller
$ ryu-manager --observe-links main.py
```

https://user-images.githubusercontent.com/53852453/134792988-7864685c-7be6-4aa7-ab9d-76fb478eaaf4.mp4


## Folder description
## Topology and data generation
1. Program of mininet topology creation
2. data generation (ping log)
3. data transfomer (ping log to TFRecord)

[Data generation tutorial](https://github.com/popo860623/thesis_code/tree/main/Topology%20and%20data_generation)

## RouteNet
1. RouteNet program

[RouteNet training tutorial](https://github.com/popo860623/thesis_code/tree/main/RouteNet)
## controller
1. SDN controller
1. main.py program is the controller entry point
2. ArpHandler is packet_in handler includes algorithms

## model
1. RouteNet pretrained model

## Datasets
1. Datasets which already transform to TFRecods


## HackMD Content
[Ryu+Mininet Installation](https://hackmd.io/3N2L8HzzQhKqw6hXQDZhcg?view)

[Data generation tutorial](https://github.com/popo860623/thesis_code/tree/main/Topology%20and%20data_generation)

[RouteNet training tutorial](https://github.com/popo860623/thesis_code/tree/main/RouteNet)


[RouteNet Demo & Evaluation](https://hackmd.io/SvDHWK8zQmagns5XAhT6DQ?view)

[Tensorboard tutorial] (https://hackmd.io/KxXGO_V8Rw6sTiCfhLeyTg?view)


