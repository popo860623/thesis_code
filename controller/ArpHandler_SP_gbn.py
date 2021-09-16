# author: ParanoiaUPC
# email: 757459307@qq.com
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu.lib.packet import ether_types
from ryu.topology import api as topo_api
from ryu.lib.packet import ipv4
from ryu.lib.packet import arp
from ryu.lib import hub

from ryu.topology import event, switches
from ryu.topology.api import get_all_switch, get_link, get_switch,get_all_link
from ryu.lib.ofp_pktinfilter import packet_in_filter, RequiredTypeFilter
from operator import itemgetter, attrgetter
import networkx as nx
import random
import numpy as np
from collections import defaultdict
# import tensorflow as tf
# from tensorflow import keras
import routenet_with_link_cap2 as rout
import os
import csv
import time
import statistics
# graph_path = '/home/hao/backup_thesis/gbnnet/gbn.txt'
graph_path = '/home/hao/backup_thesis/gbnnet/gbn_add_link_57.txt'
# graph_path = '/home/hao/backup_thesis/gbnnet/gbn_del_node_16.txt'
class ArpHandler(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        super(ArpHandler, self).__init__(*args, **kwargs)
        self.topology_api_app = self
        self.link_to_port = {}       # (src_dpid,dst_dpid)->(src_port,dst_port)
        self.link_cost = {}
        self.access_table = {}       # {(sw,port) :[host1_ip]}
        self.switch_port_table = {}  # dpip->port_num
        self.access_ports = {}       # dpid->port_num
        self.interior_ports = {}     # dpid->port_num
        self.graph = nx.DiGraph()
        self.dps = {}
        self.switches = None
        self.cap_mat = []
        self.tmp_linkbw_matrix = []
        self.linkbw_matrix = []
        self.linkbw_matrix_old = []
        self.switch_port_to_switch = defaultdict(dict)  
        self.rbw_matrix = []

        ### ML Params
        self.link_capacities = []
        self.traffic = []
        self.delay = []
        self.n_links = 0
        self.n_paths = 0
        self.n_total = 0
        self.link_indices = []
        self.path_indices = []
        self.sequ_indices = []


        # self.routenet()


        self.create_Cap_Mat()
        self.discover_thread = hub.spawn(self._discover)
            
        
        self.adjacency = defaultdict(dict)
    def _discover(self):
        while True:
            self.get_topology(None)
            self._request_stats(None)
            self.get_switch_port_to_switch_mapping()
            hub.sleep(2)

    def get_topology(self, ev):
        """
            Get topology info
        """
        # print "get topo"
        switch_list = get_all_switch(self)
        # print switch_list
        self.create_port_map(switch_list)
        self.switches = self.switch_port_table.keys()
        links = get_link(self.topology_api_app, None)
        self.create_interior_links(links)
        self.create_access_ports()
        self.get_graph()
        self.get_linkbw_matrix()

    def _request_stats(self, datapath):
        '''
        the function is to send requery to datapath
        '''
        # self.logger.debug(
        #     "send stats reques to datapath: %16x for port and flow info", datapath.id)
        switch_list = get_all_switch(self)

        for sw in switch_list:
            datapath = sw.dp
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser
            
            
            
            req = parser.OFPFlowStatsRequest(datapath)
            datapath.send_msg(req)

            req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
            datapath.send_msg(req)

    def create_port_map(self, switch_list):
        for sw in switch_list:
            dpid = sw.dp.id
            self.graph.add_node(dpid)
            self.dps[dpid] = sw.dp
            self.switch_port_table.setdefault(dpid, set())
            self.interior_ports.setdefault(dpid, set())
            self.access_ports.setdefault(dpid, set())

            for p in sw.ports:

                self.switch_port_table[dpid].add(p.port_no)
                # print('dpid = ', dpid, 'port = ', p.port_no)

    def create_interior_links(self, link_list):
        for link in link_list:
            src = link.src
            dst = link.dst
            self.link_to_port[
                (src.dpid, dst.dpid)] = (src.port_no, dst.port_no)

            # Find the access ports and interiorior ports
            if link.src.dpid in self.switches:
                self.interior_ports[link.src.dpid].add(link.src.port_no)
            if link.dst.dpid in self.switches:
                self.interior_ports[link.dst.dpid].add(link.dst.port_no)

    def create_access_ports(self):
        for sw in self.switch_port_table:
            all_port_table = self.switch_port_table[sw]
            interior_port = self.interior_ports[sw]
            self.access_ports[sw] = all_port_table - interior_port

    def create_Cap_Mat(self):
        G = nx.read_gml(graph_path, destringizer=int)
        self.cap_mat = np.full((G.number_of_nodes()+1, G.number_of_nodes()+1), fill_value=0)
        for node in range(G.number_of_nodes()):
            for adj in G[node]:
                self.cap_mat[node+1, adj+1] = int(G[node][adj][0]['bandwidth'])

        self.linkbw_matrix = np.full((len(self.cap_mat),len(self.cap_mat)),0)
        self.linkbw_matrix_old = np.full((len(self.cap_mat),len(self.cap_mat)),0)
        self.tmp_linkbw_matrix = np.full((len(self.cap_mat),len(self.cap_mat)),0)
        self.switch_port_to_switch = np.full((len(self.cap_mat),len(self.cap_mat)),0) 
        self.link_capacities = self.cap_mat[self.cap_mat != 0]
        self.n_links = len(self.link_capacities)

        print ('self.cap_mat = ', str(self.cap_mat))



    def get_graph(self):
        link_list = topo_api.get_all_link(self)
        print('self.cap_mat = ',self.cap_mat)
        print ('RBW_MATRIX = \n' + str(self.rbw_matrix)) 
        self.link_cost = {}
        for link in link_list:

            src_dpid = link.src.dpid
            dst_dpid = link.dst.dpid
            src_port = link.src.port_no
            dst_port = link.dst.port_no
            if (src_dpid, dst_dpid) not in self.link_cost.keys():
                self.link_cost[(src_dpid, dst_dpid)] = (40000.0/self.rbw_matrix[src_dpid][dst_dpid])
                self.link_cost[(dst_dpid,src_dpid)] = (40000.0/self.rbw_matrix[dst_dpid][src_dpid])                          

            self.graph.add_edge(src_dpid, dst_dpid,
                                src_port=src_port,
                                dst_port=dst_port,
                                cost=self.link_cost[(src_dpid, dst_dpid)])
        return self.graph

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath

        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)
        # print(pkt)
        eth_type = pkt.get_protocols(ethernet.ethernet)[0].ethertype
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        arp_pkt = pkt.get_protocol(arp.arp)
        ip_pkt = pkt.get_protocol(ipv4.ipv4)

        if eth_type == ether_types.ETH_TYPE_LLDP:
            # ignore lldp packet
            return
        # print('packet in.')
        if ip_pkt:
            src_ipv4 = ip_pkt.src
            src_mac = eth_pkt.src
            if src_ipv4 != '0.0.0.0' and src_ipv4 != '255.255.255.255':
                self.register_access_info(datapath.id, in_port, src_ipv4, src_mac)
            # print('ipv4')
        if arp_pkt:
            arp_src_ip = arp_pkt.src_ip
            arp_dst_ip = arp_pkt.dst_ip
            mac = arp_pkt.src_mac
            # print('arp')
            # Record the access info
            self.register_access_info(datapath.id, in_port, arp_src_ip, mac)

    def register_access_info(self, dpid, in_port, ip, mac):
        """
            Register access host info into access table.
        """
        # print "register " + ip
        if in_port in self.access_ports[dpid]:
            if (dpid, in_port) in self.access_table:
                if self.access_table[(dpid, in_port)] == (ip, mac):
                    return
                else:
                    self.access_table[(dpid, in_port)] = (ip, mac)
                    return
            else:
                self.access_table.setdefault((dpid, in_port), None)
                self.access_table[(dpid, in_port)] = (ip, mac)
                return

    def get_host_location(self, host_ip):
        """
            Get host location info:(datapath, port) according to host ip.
        """
        for key in self.access_table.keys():
            if self.access_table[key][0] == host_ip:
                return key
        self.logger.debug("%s location is not found." % host_ip)
        return None

    def get_switches(self):
        return self.switches

    def get_links(self):
        return self.link_to_port

    def get_datapath(self, dpid):
        if dpid not in self.dps:
            switch = topo_api.get_switch(self, dpid)[0]
            self.dps[dpid] = switch.dp
            return switch.dp
        return self.dps[dpid]


    def set_shortest_path(self,
                          ip_src,
                          ip_dst,
                          src_dpid, 
                          dst_dpid, 
                          to_port_no,
                          to_dst_match,
                          pre_actions=[]
                          ):
        # time.sleep(1.5)
        if nx.has_path(self.graph, src_dpid, dst_dpid):
            path = nx.shortest_path(self.graph, src_dpid, dst_dpid,weight='cost')
        else:
            path = None
        if path is None:
            self.logger.info("Get path failed.")
            return 0
        if self.get_host_location(ip_src)[0] == src_dpid:
            # paths = nx.all_shortest_paths(self.graph, src_dpid, dst_dpid)
            paths = list(nx.all_simple_paths(self.graph, src_dpid, dst_dpid))
            all_p = []
            for p in paths:
                if len(p) < 6:
                    all_p.append(p) 
            paths = all_p
            self.traffic.append(self.linkbw_matrix[src_dpid][dst_dpid])

            # print "All the shortest from " + ip_src + " to " + ip_dst + " are:"

            for spath in paths:
                tmp_cost = 0
                # print('path :' , spath)
                for i in range(len(spath)-1):
                    # print('cost = ',tmp_cost)
                    tmp_cost = tmp_cost + self.graph[spath[i]][spath[i+1]]['cost']
                # print('install path = ', path)
                print (ip_src + ' ->', spath, "-> " + ip_dst + "     cost : " + str(tmp_cost))

            # print "Shortest path from " + ip_src + " to " + ip_dst +'is:'
            # print ip_src + ' ->',
            # for sw in path:
            #     print str(sw) + ' ->',
            # print ip_dst
        if len(path) == 1:
            dp = self.get_datapath(src_dpid)
            actions = [dp.ofproto_parser.OFPActionOutput(to_port_no)]
            self.add_flow(dp, 10, to_dst_match, pre_actions+actions)
            port_no = to_port_no
        else:
            print('set sp path = ' , path)
            self.install_path(to_dst_match, path, pre_actions)
            dst_dp = self.get_datapath(dst_dpid)
            actions = [dst_dp.ofproto_parser.OFPActionOutput(to_port_no)]
            self.add_flow(dst_dp, 10, to_dst_match, pre_actions+actions)
            port_no = self.graph[path[0]][path[1]]['src_port']

        return port_no

    def install_path(self, match, path, pre_actions=[]):
        print('install path ...')
        for index, dpid in enumerate(path[:-1]):
            port_no = self.graph[path[index]][path[index + 1]]['src_port']
            dp = self.get_datapath(dpid)
            actions = [dp.ofproto_parser.OFPActionOutput(port_no)]
            # timeout = int(random.uniform(30))
            timeout=0
            self.add_flow(dp, 10, match, pre_actions+actions,timeout,timeout)

    def add_flow(self, dp, p, match, actions, idle_timeout=0, hard_timeout=0):
        ofproto = dp.ofproto
        parser = dp.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]

        mod = parser.OFPFlowMod(datapath=dp, priority=p,
                                idle_timeout=idle_timeout,
                                hard_timeout=hard_timeout,
                                match=match, instructions=inst)
        dp.send_msg(mod)
        # print('Packet Out Mes.')


    @set_ev_cls(event.EventLinkAdd, MAIN_DISPATCHER)
    def link_add_handler(self, event):
        s1 = event.link.src
        s2 = event.link.dst
        self.adjacency[s1.dpid][s2.dpid] = s1.port_no
        self.adjacency[s2.dpid][s1.dpid] = s2.port_no

    def get_linkbw_matrix(self):
        self.rbw_matrix = np.copy(self.cap_mat)
        self.linkbw_matrix_old = np.copy(self.linkbw_matrix)
        self.linkbw_matrix = np.copy(self.tmp_linkbw_matrix)
        # print('self.linkbw_matrix = ',self.linkbw_matrix)
        LINKBW_MATRIX = (self.linkbw_matrix - self.linkbw_matrix_old) / 2
        print ('LINKBW_MATRIX = \n' , LINKBW_MATRIX.astype(int))
        # print 'DELAY_MATRIX = \n' + str(np.matrix(self.delay_matrix))
        self.rbw_matrix = self.rbw_matrix - LINKBW_MATRIX
        # print ('RBW_MATRIX = \n' + str(self.rbw_matrix))
        per_link_util = []
        total_util = 0
        count = 0
        for i in range(len(LINKBW_MATRIX)):
            for j in range(i+1,len(LINKBW_MATRIX)):
                if self.cap_mat[i][j] != 0:
                    count += 1
                    tmp = ( (LINKBW_MATRIX[i][j])/ 1048576 )/ (self.cap_mat[i][j]*1.048576 / 10000 )
                    per_link_util.append(tmp)  # 單位要統一，轉為MB

        total_util =  sum(per_link_util)/len(per_link_util)
        diff = []
        for u in per_link_util:
            diff.append(abs(u-total_util))
        print('----------------------------------')
        sd = statistics.stdev(per_link_util)

        total_util = total_util if total_util < 1 else 1
        print('link utilization = ', total_util)
        if total_util > 0.01:
            ###################### Write Util ###########################
            with open('gbn_result_SP_100ms.txt','a') as f:
                f.write(str(total_util)+ '\n')
            total_bw = 0
            for i in range(len(LINKBW_MATRIX)):
                for j in range(i+1,len(LINKBW_MATRIX)):
                    total_bw += LINKBW_MATRIX[i][j]
            ###################### Write Throughput ###########################
            with open('gbn_result_SP_100ms_bw.txt','a') as f:
                f.write(str(total_bw) + '\n')
            print('Throughput = ', total_bw)
            ###################### Write Util ###########################
            # with open('link_util.csv', 'a+') as f:
            #         for dp in self.tmp_switch_port_parm:
            #             for port in self.tmp_switch_port_parm[dp]:
            #                 print('switch id : %d ,port num : %d , params : %s' %(dp,port,self.tmp_switch_port_parm[dp][port]))
            #                 print(dp,port,self.tmp_switch_port_parm[dp][port],file=f)
    def get_switch_port_to_switch_mapping(self):
        for link in get_all_link(self):
            if link.src.port_no != 4294967294:
                # print 'link.src.dpid = ' + str(link.src.dpid) + ',link.src.port = ' + str(link.src.port_no) + 'link.dst.dpid = ' + str(link.dst.dpid)
                self.switch_port_to_switch[link.src.dpid][link.src.port_no] = link.dst.dpid
                self.switch_port_to_switch[link.dst.dpid][link.dst.port_no] = link.src.dpid


    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        '''
        monitor to require the port state, then this function is to get infomation for port`s info
        '''
        switch = ev.msg.datapath
        body = ev.msg.body
        for stat in sorted(body, key=attrgetter('port_no')):
            # condition 2 means link(switch,host) = 0 which we didn't consider

            if stat.port_no != 4294967294 and self.switch_port_to_switch[switch.id][stat.port_no] != 0:
                self.tmp_linkbw_matrix[switch.id][self.switch_port_to_switch[switch.id][stat.port_no]] = stat.tx_bytes
                # self.tmp_switch_port_parm[switch.id][stat.port_no] = [stat.tx_bytes,self.cap_mat[switch.id][self.switch_port_to_switch[switch.id][stat.port_no]]]
