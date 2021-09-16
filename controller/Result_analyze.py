import re
import os
import time
# traffic_dir = '/home/hao/backup_thesis/traffic_dir/Result_Test/ML_changepath/1ms/'
traffic_dir = '/home/hao/backup_thesis/traffic_dir/new_result_test_util/ML/100ms/'
# traffic_dir = '/home/hao/backup_thesis/traffic_dir/Result_Test_gbn/QPSP/100ms/'
folder_list = os.listdir(traffic_dir)
pattern_delay = re.compile(r'time=(\d.*) ms+')
pattern_duration = re.compile(r'duration=(\d.*)')
pattern_path = re.compile(r'path = \[(.*)\]')
pattern_duration2 = re.compile(r'time (\d.*)ms')
pattern_avg_delay = re.compile(r'mdev = (\d+\.\d+\/\d+\.\d+)')

count = 0
total = 0
for folder in folder_list:
    count += 1
    f = []
    sum_ = 0
    for (dirpath, dirnames, filenames) in os.walk(traffic_dir + folder):
        f.extend(filenames)
    # print(f)
    # print('------------------ Path set %d ----------------' % count)
    
    packet_loss = []
    duration_list = []
    duration_list2 = []
    path_list = []
    for fname in f:   
        # print(traffic_dir + folder,fname)
        delay_float_list = []
        if fname == 'check.txt':
            continue
        with open(traffic_dir + folder + '/' + fname , 'r') as fi:
            content = fi.read()
            # print(content)
            delay = pattern_delay.findall(content)

            # if fname == '58.log':
            #     for x in delay:
            #         print(x)

            delay_float_list = [float(x) for x in delay]
            avg_delay_per_path = pattern_avg_delay.findall(content)
            # print('file = ', traffic_dir + folder + '/' + fname , 'a = ' , avg_delay_per_path)
            duration2 = pattern_duration2.findall(content)  #Ping回傳的Path Delay
            path = pattern_path.findall(content)
            pkt_loss = 1-(len(delay_float_list)/250.0)
            # pkt_loss = 1-(len(delay_float_list)/350.0)
            # duration_list.append(duration)
            if duration2:
                duration_list2.append(duration2[0])
            path_list.append(path)
            packet_loss.append(pkt_loss)      
            packet_loss = [float(x) for x in packet_loss] 
        # print(avg_delay_per_path[0].split('/')[1]) 
        # print(path)
        # print( traffic_dir + folder + '/' + fname, path , avg_delay_per_path[0].split('/')[1]) 
    # print('--------------------------------------------------------------')
    # for i in range(len(path_list)):
    #     print('Path : {:<16}| Duration : {:<10} | Packet loss : {:.3f}'.format(str(path_list[i]),str(duration_list2[i]),packet_loss[i]))
    duration_list2 = [float(x) for x in duration_list2]
 
    # Avg Pkt Loss
    # print(sum(packet_loss)/182)
    # print(sum(packet_loss)/272)
    # Max Pkt Loss
    print(max(packet_loss))

    # Delay
    # print(sum(duration_list2)/182)
    # print(sum(duration_list2)/272)