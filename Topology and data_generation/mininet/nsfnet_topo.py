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
import numpy as np
import time
import sys

from mininet.util import waitListening
graph_path = 'nsfnetbw/graph_attr.txt'
# graph_path = 'nsfnetbw/graph_attr_remove_03.txt' # remove (0,3)
# graph_path = 'nsfnetbw/graph_attr_remove_109.txt' # remove (10,9)
# graph_path = 'nsfnetbw/graph_attr_add_1213.txt'
# graph_path = 'nsfnetbw/graph_attr_add_313.txt'

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

        return graphs_dic

    def create_Cap_Mat(self):
        global cap_mat
        cap_mat = np.full(
            (G.number_of_nodes()+1, G.number_of_nodes()+1), fill_value=None)
        for node in range(G.number_of_nodes()):
            for adj in G[node]:
                pos = G[node][adj][0]['bandwidth'].index('bps')
                cap_mat[node, adj] = int(
                    G[node][adj][0]['bandwidth'][0:pos-1] + '000')
        # info(cap_mat)
        # all_pairs_shortest_path = [p for p in nx.all_pairs_shortest_path(G)]


def MyNetwork(net, cmd='/usr/sbin/sshd', opts='-D', ip='10.123.123.1/32', routes=None, switch=None):
    global G

    info('*** Adding controller\n')

    c0 = net.addController(name='c0',
                           controller=RemoteController,
                           protocol='tcp',
                           port=6633, ip='127.0.0.1')

    info('*** Adding host\n')
    for n in G.nodes():
        net.addSwitch("s%s" % str(n+1))
        if int(n) in list(G.nodes()):
            net.addHost('h%s' % str(n+1), ip='10.0.0.' + str(n+1))
            net.addLink('s%s' % str(n+1), 'h%s' % str(n+1))
            info('h%s ' % str(n+1))

    existed_edge = []
    info(G.edges())

    for (n1, n2) in G.edges():
        if (n1, n2) not in existed_edge:
            info((n1, n2))
            net.addLink('s%s' % str(n1+1), 's%s' % str(n2+1),
                        cls=TCLink, bw=(cap_mat[n1][n2]/10000)*8)
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

    # info( '*** Post configure switches and hosts\n')

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


def swap(list_, a, b):
    t = list_[a]
    list_[a] = list_[b]
    list_[b] = t
    return list_


def generate_flow(net, n_files, G):
    hosts = net.hosts
    all_pairs_shortest_path = []
    flow_count = 0

    # Generate All Path
    for i in range(14):
        for j in range(14):
            if i != j:
                all_pairs_shortest_path.append(nx.shortest_path(G, i, j))

    while n_files:
        if n_files == 3800:
            break
        traffic_dir = 'traffic_dir/new_result_test_util/RPSP/10ms/'
        if not os.path.exists(traffic_dir + str(n_files)):
            os.makedirs(traffic_dir + str(n_files))
            traffic_dir += str(n_files)
        for i in range(182, 0, -1):
            time.sleep(0.001)
            # Random Select Path
            idx = random.randint(0, i-1)
            path = all_pairs_shortest_path[idx]
            # path = all_pairs_shortest_path[i-1]
            src_h = "h" + str(path[0]+1)
            dst_h = "h" + str(path[1]+1)
            # print('h1 = ', src_h,"h2 = ", dst_h)
            src = net.get(src_h)
            dst = net.get(dst_h)

            # create cmd ping
            server_cmd = "START_TIME=$(date +%s%3N) && "  # get begin time
            pkt_size = 1000
            server_cmd += "ping -c 250 -s " + \
                str(pkt_size) + " -i 0.01 " + dst.IP()
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
                  )
    argvopts = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else (
        '-D -o UseDNS=no -u0')

    MyNetwork(net, opts=argvopts)
