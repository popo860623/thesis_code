# -*- coding: utf-8 -*
from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSController
from mininet.node import CPULimitedHost, Host, Node
from mininet.node import OVSKernelSwitch, UserSwitch
from mininet.node import IVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink, Intf
import networkx as nx
import random
import os
import tarfile
import traceback
import numpy as np
import csv
import re
import threading
import time
import matplotlib.pyplot as plt
# graph_path = 'gbnnet/gbn.txt'
graph_path = 'gbnnet/gbn_add_link_57.txt'
traffic_file = 'output_data.csv'
G = nx.Graph()

cat_mat = []

path_set = []
traffic_list = []
all_pairs_shortest_path = []


class DataNet:
    def generate_graphs_dic(self):
        global G

        graphs_dic = {}
        G = nx.read_gml(graph_path, destringizer=int)
        graphs_dic[graph_path] = G
        # nx.draw_networkx(G,node_color='green',with_label=1)
        # plt.show()
        return graphs_dic

    def create_Cap_Mat(self):
        global cap_mat

        cap_mat = np.full(
            (G.number_of_nodes()+1, G.number_of_nodes()+1), fill_value=None)
        for node in range(G.number_of_nodes()):
            for adj in G[node]:
                cap_mat[node, adj] = int(G[node][adj][0]['bandwidth'])
        info(cap_mat)
        # all_pairs_shortest_path = [p for p in nx.all_pairs_shortest_path(G)]

    def generate_traffic(self):
        global traffic_list
        for i in range(G.number_of_nodes()):
            for j in range(G.number_of_nodes()):
                if i != j:
                    path_set.append(nx.shortest_path(G, i, j))

        with open(traffic_file) as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                traffic_list.append(row[1])
        # print(path_set)
        # for x,y in zip(path_set,traffic_list):
        #     print('path = ',x , 'bws = ', y)


def MyNetwork(net):
    global G

    info('*** Adding controller\n')
    c0 = net.addController(name='c0',
                           controller=RemoteController,
                           protocol='tcp',
                           port=6633, ip='127.0.0.1')

    for n in G.nodes():
        net.addSwitch("s%s" % str(n+1))
        if int(n) in list(G.nodes()):
            net.addHost('h%s' % str(n+1), ip='10.0.0.' + str(n+1))
            net.addLink('s%s' % str(n+1), 'h%s' % str(n+1))
    existed_edge = []
    for (n1, n2) in G.edges():
        if (n1, n2) not in existed_edge:
            info((n1, n2))
            net.addLink('s%s' % str(n1+1), 's%s' % str(n2+1),
                        cls=TCLink, bw=cap_mat[n1][n2]/10000*8)
            existed_edge.append((n1, n2))
            existed_edge.append((n2, n1))

    info('*** Starting network\n')
    net.build()

    info('*** Starting controllers\n')
    for controller in net.controllers:
        controller.start()

    info('*** Starting switches\n')
    for n in range(0, G.number_of_nodes()):
        net.get('s%s' % str(n+1)).start([c0])

    info('*** Post configure switches and hosts\n')
    while True:
        try:
            user_input = raw_input("GEN/CLI/QUIT : ")
        except EOFError as error:
            user_input = "QUIT"

        if user_input.upper() == "GEN":
            generate_flow(net, 50, G)

        elif user_input.upper() == "CLI":
            info("Running CLI...\n")
            CLI(net)
        elif user_input.upper() == "QUIT":
            info("Terminating...\n")
            net.stop()
            break
        else:
            print("Command not found")


def swap(list, a, b):
    t = list[a]
    list[a] = list[b]
    list[b] = t
    return list


def generate_flow(net, n_files, G):
    hosts = net.hosts
    all_pairs_shortest_path = []
    flow_count = 0
    # Generate All Path
    for i in range(17):
        for j in range(17):
            if i != j:
                all_pairs_shortest_path.append(nx.shortest_path(G, i, j))


    while(n_files):
        if n_files == 200:
            break
        traffic_dir = 'traffic_dir/Result_Test_gbn/addlink57/'
        if not os.path.exists(traffic_dir + str(n_files)):
            os.makedirs(traffic_dir + str(n_files))
            traffic_dir += str(n_files)
        for i in range(272, 0, -1):
            time.sleep(0.001)
            # Random Select Path
            idx = random.randint(0, i-1)
            path = all_pairs_shortest_path[idx]
            src_h = "h" + str(path[0]+1)
            dst_h = "h" + str(path[1]+1)

            src = net.get(src_h)
            dst = net.get(dst_h)

            # create cmd ping
            server_cmd = "START_TIME=$(date +%s%3N) && "  # get begin time
            pkt_size = 1000
            server_cmd += "ping -c 350 -s " + \
                str(pkt_size) + " -i 0.1 " + dst.IP()
            server_cmd += " > " + traffic_dir + "/" + str(i) + ".log && "
            server_cmd += "echo path = " + \
                str(path) + " >> " + traffic_dir + "/" + str(i) + ".log && "
            server_cmd += "ELAPSED_TIME=$(($(date +%s%3N)-$START_TIME)) && "
            server_cmd += "echo duration=$ELAPSED_TIME >> " + \
                traffic_dir + "/" + str(i) + ".log &"
            # print(server_cmd)
            # send the cmd
            src.cmdPrint(server_cmd)
            # src.popen(server_cmd,shell=True)
            all_pairs_shortest_path = swap(all_pairs_shortest_path, idx, i-1)
        time.sleep(1)
        n_files -= 1


if __name__ == "__main__":
    setLogLevel('info')
    datanet = DataNet()
    datanet.generate_graphs_dic()
    datanet.create_Cap_Mat()


    net = Mininet(topo=None,
                  build=False,
                  ipBase='10.0.0.0/8')

    MyNetwork(net)
