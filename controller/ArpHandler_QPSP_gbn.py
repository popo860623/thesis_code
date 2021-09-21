# author: ParanoiaUPC
# email: 757459307@qq.com
import re
import statistics
import time
import routenet_with_link_cap2 as rout
import tensorflow as tf
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
from ryu.topology.api import get_all_switch, get_link, get_switch, get_all_link
from ryu.lib.ofp_pktinfilter import packet_in_filter, RequiredTypeFilter
from operator import itemgetter, attrgetter
import networkx as nx
import random
import numpy as np
from collections import defaultdict
import sys
sys.path.append('/home/hao/.local/lib/python3.6/site-packages')
sys.path.append('/usr/lib/python3.6')
graphPredict = tf.Graph()
graph_path = '/home/hao/thesis_code/Topology and data_generation/NetworkX_graph_file/gbnnet/gbn.txt'
G = nx.Graph()
G = nx.read_gml(graph_path, destringizer=int)


class ArpHandler(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    #####################################################################################

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
        self.switch_port_to_switch = []
        self.input_rate_matrix = []
        self.rbw_matrix = []
        self.linkbw_matrix_now = []
        # ML Params
        self.link_capacities = []
        self.traffic = []
        self.jitter = []
        self.delay = []
        self.n_links = 0
        self.n_paths = 0
        self.n_total = 0
        self.link_indices = []
        self.path_indices = []
        self.sequ_indices = []
        self.built = 0
        self.create_Cap_Mat()
        self.discover_thread = hub.spawn(self._discover)
        self.predict_thrad = hub.spawn(self._predict_thread)
        self.all_pairs_shortest_path = []
        self.get_all_path()
        self.adjacency = defaultdict(dict)
        self.model = None
        self.prediction = None
        self.all_path = []
        self.pkt_count = 0
        self.global_time = time.time()
    #####################################################################################

    def init_all_feature(self):
        self.traffic = []
        self.delay = []
        self.n_links = 0
        self.n_paths = 0
        self.n_total = 0
        self.link_indices = []
        self.path_indices = []
        self.sequ_indices = []
        self.all_pairs_shortest_path = []

    def get_all_path(self):
        for i in range(14):
            for j in range(14):
                if i != j:
                    self.all_pairs_shortest_path.append(
                        nx.shortest_path(G, i, j, weight='cost'))

        for i in range(len(self.all_pairs_shortest_path)):
            self.all_pairs_shortest_path[i] = [
                x+1 for x in self.all_pairs_shortest_path[i]]
        # print('all path  =  ', self.all_pairs_shortest_path)

    def _predict_thread(self):
        while(1):
            now_time = time.time()
            if(now_time - self.global_time) >= 5:
                print('Ready for prediction')
                self.predictor()
                print('Predict Complete. Then Init All Features...')
                self.init_all_feature()
                print('Intialization Done.')
                self.get_all_path()
                self.global_time = now_time
                break
            print('wait .......')
            hub.sleep(1)

    def _discover(self):
        while True:
            self._request_stats(None)
            self.get_topology(None)
            self.get_switch_port_to_switch_mapping()
            hub.sleep(2)

    def get_topology(self, ev):
        """
            Get topology info
        """
        switch_list = get_all_switch(self)
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
            self.link_to_port[(src.dpid, dst.dpid)] = (
                src.port_no, dst.port_no)

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

    #####################################################################################
    def create_Cap_Mat(self):
        G = nx.read_gml(graph_path, destringizer=int)
        self.cap_mat = np.full(
            (G.number_of_nodes()+1, G.number_of_nodes()+1), fill_value=0)
        for node in range(G.number_of_nodes()):
            for adj in G[node]:
                self.cap_mat[node+1, adj+1] = int(G[node][adj][0]['bandwidth'])
        self.linkbw_matrix = np.full((len(self.cap_mat), len(self.cap_mat)), 0)
        self.linkbw_matrix_old = np.full(
            (len(self.cap_mat), len(self.cap_mat)), 0)
        self.tmp_linkbw_matrix = np.full(
            (len(self.cap_mat), len(self.cap_mat)), 0)
        self.switch_port_to_switch = np.full(
            (len(self.cap_mat), len(self.cap_mat)), 0)
        self.link_capacities = self.cap_mat[self.cap_mat != 0]
        self.n_links = len(self.link_capacities)
        self.input_rate_matrix = np.full((G.number_of_nodes()+1,8),0)
    #####################################################################################

    #####################################################################################
    def get_graph(self):
        link_list = topo_api.get_all_link(self)
        self.link_cost = {}
        for link in link_list:
            src_dpid = link.src.dpid
            dst_dpid = link.dst.dpid
            src_port = link.src.port_no
            dst_port = link.dst.port_no
            if (src_dpid, dst_dpid) not in self.link_cost.keys():
                self.link_cost[(src_dpid, dst_dpid)] = (
                    40000/self.rbw_matrix[src_dpid][dst_dpid])
                self.link_cost[(dst_dpid, src_dpid)] = (
                    40000/self.rbw_matrix[dst_dpid][src_dpid])
            self.graph.add_edge(src_dpid, dst_dpid,
                                src_port=src_port,
                                dst_port=dst_port,
                                cost=self.link_cost[(src_dpid, dst_dpid)])
        return self.graph
    #####################################################################################

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath

        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        pkt = packet.Packet(msg.data)

        eth_type = pkt.get_protocols(ethernet.ethernet)[0].ethertype
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        arp_pkt = pkt.get_protocol(arp.arp)
        ip_pkt = pkt.get_protocol(ipv4.ipv4)

        if eth_type == ether_types.ETH_TYPE_LLDP:
            return

        if ip_pkt:
            src_ipv4 = ip_pkt.src
            src_mac = eth_pkt.src
            if src_ipv4 != '0.0.0.0' and src_ipv4 != '255.255.255.255':
                self.register_access_info(
                    datapath.id, in_port, src_ipv4, src_mac)

        if arp_pkt:
            arp_src_ip = arp_pkt.src_ip
            arp_dst_ip = arp_pkt.dst_ip
            mac = arp_pkt.src_mac

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

    #####################################################################################
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
    #####################################################################################

    def parse(self, serialized, target='delay'):
        with tf.device("/cpu:0"):
            with tf.name_scope('parse'):
                features = tf.parse_single_example(
                    serialized,
                    features={
                        'traffic': tf.VarLenFeature(tf.float32),
                        target: tf.VarLenFeature(tf.float32),
                        'link_capacity': tf.VarLenFeature(tf.float32),
                        'links': tf.VarLenFeature(tf.int64),
                        'paths': tf.VarLenFeature(tf.int64),
                        'sequences': tf.VarLenFeature(tf.int64),
                        'n_links': tf.FixedLenFeature([], tf.int64),
                        'n_paths': tf.FixedLenFeature([], tf.int64),
                        'n_total': tf.FixedLenFeature([], tf.int64)
                    })
                for k in ['traffic', target, 'link_capacity', 'links', 'paths', 'sequences']:
                    features[k] = tf.sparse_tensor_to_dense(features[k])
                # for k, v in features.items():
                #     print({k:v})
        return {k: v for k, v in features.items() if k is not target}, features[target]

    def read_dataset(self):
        ds = tf.data.TFRecordDataset('./tfrecords/tf_test.tfrecords')

        ds = ds.map(lambda buf: self.parse(buf))

        ds = ds.batch(1)

        it = ds.make_initializable_iterator()
        return it

    def predictor(self, load=0):
        path_pattern = re.compile(r'path= \[(.*)\]')
        weight_pattern = re.compile(r'weight= (.*)')
        with open('../../opt_path_SP_100ms.txt') as f:
            content = f.read()
            paths = path_pattern.findall(content)
            weights = weight_pattern.findall(content)
        weights = [int(float(x)) for x in weights]
        path_list = []
        for p in paths:
            p = p[1:-1]
            path = p.split(',')
            pp = list(map(int, path))

            pp = [x+1 for x in pp]
            path_list.append(pp)

        for i in range(len(path_list)):
            ip_dst = '10.0.0.' + str(path_list[i][-1])
            datapath = self.get_datapath(path_list[i][0])
            parser = datapath.ofproto_parser
            to_dst_match = parser.OFPMatch(eth_type=2048, ipv4_dst=ip_dst)
            qos0 = max(weights)/3
            qos1 = max(weights)/3*2

            if weights[i] < qos0:
                priority = 100
            elif weights[i] > qos0 and weights[i] < qos1:
                priority = 50
            # elif weights[i] > qos1 and weights[i] < qos2:
            #     priority = 60    
            # elif weights[i] > qos2 and weights[i] < qos3:
            #     priority = 40
            else:
                priority = 10
            # priority = 10
            self.install_path(to_dst_match, path_list[i], [], priority)

        print('Flow had benn updated.')

    def set_shortest_path(self,
                          ip_src,
                          ip_dst,
                          src_dpid,
                          dst_dpid,
                          to_port_no,
                          to_dst_match,
                          pre_actions=[]
                          ):

        print('------------------------------------')

        if nx.has_path(self.graph, src_dpid, dst_dpid):
            path = nx.shortest_path(self.graph, src_dpid, dst_dpid)
        else:
            path = None
        if path is None:
            self.logger.info("Get path failed.")
            return

        if self.get_host_location(ip_src)[0] == src_dpid:
            paths = nx.all_shortest_paths(self.graph, src_dpid, dst_dpid)

            # for i in self.graph.nodes():
            #     for j in self.graph.nodes():
            #         if i!=j and len(self.all_path)<182:
            #             sp = nx.shortest_path(self.graph,i,j)
            #             if sp not in self.all_path:
            #                 self.all_path.append(sp)
            # print('all = ', self.all_path, 'len = ', len(self.all_path))

            print('src = ', src_dpid, 'dst = ', dst_dpid)
            # print ('now matrix = ' , self.linkbw_matrix_now)

            # print ('*****************************')
            # print ('traffic = ', self.traffic)
            # print ('delay = ', self.delay)
            # print ('link_cap = ', self.link_capacities)
            # print ('path = ' , self.path_indices)
            # print ('link = ', self.link_indices)
            # print ('sequ = ', self.sequ_indices)
            # print ('n_paths = ' , self.n_paths)
            # print ('n_links = ', self.n_links)
            # print ('n_total = ', self.n_total)
            # print ('*****************************')
            # print ("All the shortest from " + ip_src + " to " + ip_dst + " are:")
            # paths = nx.all_shortest_paths(self.graph, src_dpid, dst_dpid)

            # self.make_indices(src_dpid, dst_dpid)
            # self.make_tfrecord()
            # path_delay = self.predictor()

            # print('path delay = ', path_delay)
            # for spath in paths:
            #     tmp_cost = 0
            #     for i in range(len(spath)-1):
            #         tmp_cost = tmp_cost + self.graph[spath[i]][spath[i+1]]['cost']
            #         # print path[i], path[i+1], self.graph[path[i]][path[i+1]]['delay']
            #     print (ip_src , ' -> ',spath, " -> ", ip_dst, "   cost : ", str(tmp_cost))

            # print ("Shortest path from " , ip_src , " to " , ip_dst ,'is:',end='')
            # print (ip_src , ' ->',end='')
            # for sw in path:
            #     print (str(sw) , ' -> ',end='')
            # print (ip_dst)

        # print('# of path = ' , 'src = ,' ,src_dpid , 'dst = ', dst_dpid, len(self.all_path))

        if len(path) == 1:
            dp = self.get_datapath(src_dpid)
            actions = [dp.ofproto_parser.OFPActionOutput(to_port_no)]
            self.add_flow(dp, 10, to_dst_match, pre_actions+actions)
            port_no = to_port_no
        else:
            self.install_path(to_dst_match, path, pre_actions)
            dst_dp = self.get_datapath(dst_dpid)
            actions = [dst_dp.ofproto_parser.OFPActionOutput(to_port_no)]
            self.add_flow(dst_dp, 10, to_dst_match, pre_actions+actions)
            port_no = self.graph[path[0]][path[1]]['src_port']

        return port_no

    def _int64_feature(self, value):
        return tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=[value]))

    def _int64_features(self, value):
        return tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=value))

    def _float_features(self, value):
        return tf.compat.v1.train.Feature(float_list=tf.compat.v1.train.FloatList(value=value))

    def normalization(self, data):
        max_val = max(data)
        min_val = min(data)
        tmp = []

        for x in data:
            if max_val-min_val == 0:
                tmp.append(float(1/len(data)))
            else:
                tmp.append((x-min_val)/(max_val-min_val))
        return tmp

    def make_tfrecord(self):
        print("Make TFRecord.")
        writer = tf.python_io.TFRecordWriter('./tfrecords/tf_test.tfrecords')

        self.traffic += [0.] * (182-len(self.traffic))
        self.delay += [0.] * (182-len(self.delay))
        self.traffic = self.normalization(self.traffic)
        self.link_capacities = self.normalization(self.link_capacities)

        # print ('traffic = ', self.traffic)
        # print ('delay = ', self.delay)
        # print ('link_cap = ', self.link_capacities)
        # print ('path = ' , self.path_indices)
        # print ('link = ', self.link_indices)
        # print ('sequ = ', self.sequ_indices)
        # print ('n_paths = ' , self.n_paths)
        # print ('n_links = ', self.n_links)
        # print ('n_total = ', self.n_total)

        example = tf.compat.v1.train.Example(features=tf.compat.v1.train.Features(feature={
            'traffic': self._float_features(np.array(self.traffic)),
            'delay': self._float_features(np.array(self.delay)),
            'link_capacity': self._float_features(list(self.link_capacities)),
            'links': self._int64_features(self.link_indices),
            'paths': self._int64_features(self.path_indices),
            'sequences': self._int64_features(self.sequ_indices),
            'n_links': self._int64_feature(self.n_links),
            'n_paths': self._int64_feature(self.n_paths),
            'n_total': self._int64_feature(self.n_total)
        }
        ))
        example.SerializeToString()

        writer.write(example.SerializeToString())

        writer.close()

    def make_indices(self, src, dst):
        # print('src = ',src -1,'dst = ',dst-1)
        self.path_indices = []
        self.link_indices = []
        self.sequ_indices = []
        self.traffic = []
        self.delay = []
        paths = nx.all_shortest_paths(G, src, dst)

        edges = list(self.graph.edges())
        path_with_link_index = []
        for spath in self.all_pairs_shortest_path:
            tmp_path = []
            for i in range(len(spath)-1):
                tmp_path.append(edges.index((spath[i], spath[i+1])))
            path_with_link_index.append(tmp_path)
            self.traffic.append(1000)
            self.delay.append(1)
            # break
        seg = 0
        for path in path_with_link_index:
            self.link_indices += path
            self.path_indices += len(path) * [seg]
            self.sequ_indices += list(range(len(path)))
            seg += 1
        # print('link_indices ', self.link_indices,'len = ', len(self.link_indices))
        # print('sequ_indices ', self.sequ_indices,'len = ', len(self.sequ_indices))

        self.n_paths = len(path_with_link_index)
        self.n_total = len(self.path_indices)
        self.n_links = len(edges)

    def install_path(self, match, path, pre_actions=[], priority=1):

        for index, dpid in enumerate(path[:-1]):
            port_no = self.graph[path[index]][path[index + 1]]['src_port']
            dp = self.get_datapath(dpid)
            actions = [dp.ofproto_parser.OFPActionOutput(port_no)]

            print('install path = ', path, 'match = ', match, 'priority = ',
                  priority, 'len = ', len(self.all_pairs_shortest_path))
            self.add_flow(dp, priority, match, pre_actions+actions)

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

    @set_ev_cls(event.EventLinkAdd, MAIN_DISPATCHER)
    def link_add_handler(self, event):
        s1 = event.link.src
        s2 = event.link.dst
        self.adjacency[s1.dpid][s2.dpid] = s1.port_no
        self.adjacency[s2.dpid][s1.dpid] = s2.port_no

    def get_linkbw_matrix(self):
        time.sleep(0.1)
        self.rbw_matrix = np.copy(self.cap_mat)
        self.linkbw_matrix_old = np.copy(self.linkbw_matrix)
        self.linkbw_matrix = np.copy(self.tmp_linkbw_matrix)
        # print('cap = ', self.cap_mat)
        # print('self.linkbw_matrix = ',self.linkbw_matrix)
        self.linkbw_matrix_now = (
            self.linkbw_matrix - self.linkbw_matrix_old) / 2
        print('LINKBW_MATRIX = \n', self.linkbw_matrix_now)
        print('input rate = ' ,((1024/self.linkbw_matrix_now)))
        # print('input rate = \n', self.input_rate_matrix)
        # print 'DELAY_MATRIX = \n' + str(np.matrix(self.delay_matrix))
        self.rbw_matrix = self.rbw_matrix - self.linkbw_matrix_now
        # print 'RBW_MATRIX = \n' + str(self.rbw_matrix)
        per_link_util = []
        total_util = 0
        count = 0
        for i in range(len(self.linkbw_matrix)):
            for j in range(i+1, len(self.linkbw_matrix)):
                if self.cap_mat[i][j] != 0:
                    count += 1
                    tmp = (
                        (self.linkbw_matrix_now[i][j]/1048576)) / (self.cap_mat[i][j]*1.048576 / 10000)
                    per_link_util.append(tmp)  # 單位要統一，轉為MB
        total_util = sum(per_link_util)/len(per_link_util)  # Avg
        diff = []
        for u in per_link_util:
            diff.append(abs(u-total_util))
        print('--------------------------------------------------')
        sd = statistics.stdev(per_link_util)
        print('SD : ', sd)
        print('link utilization = ', total_util)
        if total_util > 0.01:
            ###################### Write Util ###########################
            with open('gbn_result_QPSP_100ms.txt', 'a') as f:
                f.write(str(total_util) + '\n')
            total_bw = 0
            for i in range(len(self.linkbw_matrix_now)):
                for j in range(i+1, len(self.linkbw_matrix_now)):
                    total_bw += self.linkbw_matrix_now[i][j]
            ###################### Write Throughput ###########################
            with open('gbn_result_QPSP_100ms_bw.txt', 'a') as f:
                f.write(str(total_bw) + '\n')
            print('Throughput = ', total_bw)

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
                self.tmp_linkbw_matrix[switch.id][self.switch_port_to_switch[switch.id]
                                                  [stat.port_no]] = stat.tx_bytes
                self.input_rate_matrix[switch.id][stat.port_no] = stat.tx_bytes
                # print('test', self.tmp_linkbw_matrix)
                # print('input rate = \n' , self.input_rate_matrix)

            
