from os import walk
import re
import networkx as nx
import numpy as np
import time
import tensorflow as tf
import os
import glob
import random
paths = []
traffics = []
delays = []
# op = 'gbn_per_200_1KB_1ms_c350/'
# op = 'gbn_del_node_16/'
op='Result_Test_gbn/addlink57/'
f = []


folder_list = os.listdir('/home/hao/backup_thesis/traffic_dir/' + op)


def _int64_feature(value):
        return tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=[value]))

def _int64_features(value):
    return tf.compat.v1.train.Feature(int64_list=tf.compat.v1.train.Int64List(value=value))


def _float_features(value):
    return tf.compat.v1.train.Feature(float_list=tf.compat.v1.train.FloatList(value=value))



graph = nx.read_gml('gbnnet/gbn_add_link_57.txt',destringizer=int)

# graph = nx.read_gml('gbnnet/gbn_del_node_16.txt',destringizer=int)

def make_indices(paths):
    path_indices=[]
    link_indices=[]
    sequ_indices=[]
    edges = list(graph.edges())
    path_with_link_index = []
    # print(paths)
    for spath in paths:
        tmp_path = []
        for i in range(len(spath)-1):
            tmp_path.append(edges.index( (spath[i],spath[i+1]) )) #gbn
 
        path_with_link_index.append(tmp_path)

    seg = 0
    for path in path_with_link_index:
            link_indices += path
            path_indices += len(path) * [seg]
            sequ_indices += list(range(len(path)))
            seg += 1
    # print('path_indices ', path_indices)
    # print('link_indices ', link_indices)
    # print('sequ_indices ', sequ_indices)
    n_paths = len(path_with_link_index)
    n_total = len(path_indices)
    n_links = len(edges)
    return path_indices,link_indices,sequ_indices,n_paths,n_total,n_links

def normalization(data):
    max_val = max(data)
    min_val = min(data)
    tmp = []

    for x in data:
        if max_val-min_val == 0:
            tmp.append(float(1/len(data)))
        else:
            tmp.append((x-min_val)/(max_val-min_val))
    return tmp

def make_tfrecord(paths=[],delays=[],traffics=[],packet_loss=[],i=0):
    
    cap_mat = np.full((graph.number_of_nodes()+1, graph.number_of_nodes()+1), fill_value=0)

    for node in range(graph.number_of_nodes()):
        for adj in graph[node]:
            cap_mat[node+1, adj+1] = int(graph[node][adj][0]['bandwidth']) # gbn
    link_capacities = cap_mat[cap_mat != 0]
    link_capacities = link_capacities * 1.E-4
    # time.sleep(0.001)

    if not os.path.exists('./topology/tfrecords/' + op):
        os.makedirs('./topology/tfrecords/' + op)
    writer = tf.io.TFRecordWriter('./topology/tfrecords/' + op + str(i) + '.tfrecords')

    # if not os.path.exists('./topology/tfrecords/tmp'):
    #     os.makedirs('./topology/tfrecords/tmp')
    # writer = tf.io.TFRecordWriter('./topology/tfrecords/tmp/'  + str(i) + '.tfrecords')

    if len(paths) > 0:
        # print('paths = ', paths)
        path_indices,link_indices,sequ_indices,n_paths,n_total,n_links = make_indices(paths)
        feature={
            'traffic': _float_features(np.array(traffics)),
            'delay': _float_features(np.array(delays)),
            'loss':_float_features(np.array(packet_loss)),
            'link_capacity': _float_features(list(link_capacities)),
            'links': _int64_features(link_indices),
            'paths': _int64_features(path_indices),
            'sequences': _int64_features(sequ_indices),
            'n_links': _int64_feature(n_links),
            'n_paths': _int64_feature(n_paths),
            'n_total': _int64_feature(n_total)
        }
   
        example = tf.compat.v1.train.Example(features=tf.compat.v1.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        writer.close()
        print('[' + str(i) + '/' + str(4000-i) + ']' + ' TFRecord ' + str(i) +  ' Done. > ' + './topology/tfrecords/' + op + str(i) + '.tfrecords' )

def split_tfrecords():
    # print('********************************************')
    tfr_train = '/home/hao/backup_thesis/topology/tfrecords/' + 'train_gbn_per_200_1KB_1ms_c350/'
    tfr_evaluate = '/home/hao/backup_thesis/topology/tfrecords/' + 'eval_gbn_per_200_1KB_1ms_c350/'
    tfr_test = '/home/hao/backup_thesis/topology/tfrecords/' + 'test_gbn_per_200_1KB_1ms_c350/'
    if not os.path.exists(tfr_train):
        os.makedirs(tfr_train)
    if not os.path.exists(tfr_evaluate):
        os.makedirs(tfr_evaluate)
    if not os.path.exists(tfr_test):
        os.makedirs(tfr_test)
    
    tfrecords = glob.glob('./topology/tfrecords/tmp/' + '*.tfrecords')
    # tfrecords = glob.glob('./topology/tfrecords/traffic_dir/' + op + '*.tfrecords')
    training = len(tfrecords) * 0.8
    print('Num of training data = ', len(tfrecords))

    train_samples = random.sample(tfrecords, int(training))         # Train

    evaluate_samples = list(set(tfrecords) - set(train_samples))    
    evaluate_num = len(evaluate_samples)*0.9
    eval_samples = random.sample(set(evaluate_samples),int(evaluate_num))   # Eval

    test_samples = list(set(evaluate_samples) - set(eval_samples))      # Test
    for file in train_samples:
        file_name = file.split('/')[-1]
        os.rename(file, tfr_train + file_name)

    for file in eval_samples:
        file_name = file.split('/')[-1]
        os.rename(file, tfr_evaluate + file_name)

    for file in test_samples:
        file_name = file.split('/')[-1]
        os.rename(file, tfr_test + file_name)
# Regular Expression
#############################################################################
i=0
pattern_delay = re.compile(r'time=(\d.*) ms+')
pattern_duration = re.compile(r'duration=(\d.*)')
pattern_traffic = re.compile(r'(\d+) bytes')
pattern_path = re.compile(r'path = \[(.*)\]')
pattern_avg_delay = re.compile(r'mdev = (\d+\.\d+\/\d+\.\d+)')
pattern_delay = re.compile(r'time=(\d+.\d+)')
pattern_max = re.compile(r'mdev = (\d+\.\d+\/\d+\.\d+\/\d+\.\d+)')
#############################################################################

for folder in folder_list:
    f = []
    # print('Making Folder ', './traffic_dir/' + op + folder + ' to TFRecodes.... ')
    # for (dirpath, dirnames, filenames) in walk('./traffic_dir/Result_Test/Changepath/' + op + folder):
    # change path nsfnet
    # for (dirpath, dirnames, filenames) in walk('./ChangePath/train_test2/' + folder):
    #     f.extend(filenames)
    #gbn
    for (dirpath, dirnames, filenames) in walk('./traffic_dir/' + op + folder):
        f.extend(filenames)
    paths = []
    delays = []
    traffics = []
    packet_loss = []
    for fname in f:         
        with open('./traffic_dir/' + op + folder  + "/" + fname) as fi: 
            content = fi.read()
            delay = pattern_delay.findall(content)
            delay = [float(x) for x in delay]
            avg_delay = pattern_avg_delay.findall(content)
            ################################################################################
            duration = pattern_duration.findall(content)    #手動算的Path Delay
            if duration:
                duration = [d[:-1] for d in duration]   # remove N in last char
                duration = [int(d) for d in duration]   # conver str list to int list
                duration_max = max(duration)            # get last packet receive time
                path_dur = duration_max - duration[0]
            #################################################################################
            
            traffic = pattern_traffic.findall(content)
            path = pattern_path.findall(content)

            if len(delay) > 0:
                if avg_delay:
                    delay = avg_delay[0].split('/')[1]
                else:
                    delay = sum(delay)/len(delay)
                # pkt_loss = len(delay) / 250.0
                # packet_loss.append(pkt_loss) 
                delays.append(float(delay))
                traffics.append(float(traffic[0]))
                # path = ['1,2,3,4'] , path[0] = 1,2,3,4 -->str type
                if path:
                    path = path[0].split(', ')
                else:
                    print('./traffic_dir/' + op + folder  + "/" + fname)
                    break
                path = list(map(int,path)) # using map and convert to list
                paths.append(path)
    print(len(paths))
    if len(traffics) > 0:
        traffics = [x * 1E-3 for x in traffics]
        # delays = normalization(delays)

    i+=1
    make_tfrecord(paths,delays,traffics,packet_loss,i)
# split_tfrecords()