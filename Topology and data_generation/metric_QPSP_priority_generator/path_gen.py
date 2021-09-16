########################################################################################
# Caculate path's weight and generate data for metric-based algorithm(metric、QPSP)
########################################################################################
import re
import os
import time

# traffic_dir = '/home/hao/backup_thesis/traffic_dir/Result_Test_gbn/SP/100ms/' # 根據input rate來更改traffic dir
traffic_dir = '/home/hao/backup_thesis/traffic_dir/new_result_test_util/100ms/' 
folder_list = os.listdir(traffic_dir)
pattern_delay = re.compile(r'time=(\d.*) ms+')                          # Regular Expression for packet delay
pattern_duration = re.compile(r'duration=(\d.*)')                       # Regular Expression for path delay
pattern_path = re.compile(r'path = \[(.*)\]')                           # Regular Expression for path
pattern_duration2 = re.compile(r'time (\d.*)ms')                        # Regular Expression for packet delay
count = 0
total = 0

for folder in folder_list:
    if count >= 1:
        break
    count += 1
    f = []
    avg_delay_per_folder = []
    for (dirpath, dirnames, filenames) in os.walk(traffic_dir + folder):
        f.extend(filenames)

    ########################################################################
    packet_loss = []
    duration_list = []
    duration_list2 = []
    path_list = []
    weight_list = []
    avg_delay = []
    ########################################################################

    for fname in f:   
        # print(traffic_dir + folder,fname)
        delay_float_list = []
        with open(traffic_dir + folder + '/' + fname , 'r') as fi:
            content = fi.read()
            delay = pattern_delay.findall(content)
            delay_float_list = [float(x) for x in delay]
  
            duration2 = pattern_duration2.findall(content)  #Ping回傳的Path Delay
            path = pattern_path.findall(content)
            pkt_loss = 1-(len(delay_float_list)/250.0)      # NSF 250 packets
            # pkt_loss = 1-(len(delay_float_list)/350.0)    # GBN 350 packets
            avg_delay.append(sum(delay_float_list)/250.0)
            # avg_delay.append(sum(delay_float_list)/350.0)
            if duration2:
                duration_list2.append(duration2[0])
            path_list.append(path)
            packet_loss.append(pkt_loss)      
            packet_loss = [float(x) for x in packet_loss]    # packet loss per path
            print(delay_float_list, len(delay_float_list))
    # Write to file
    with open('opt_path_SP_100ms.txt', 'a+') as f:
        for i in range(len(path_list)):
            print('Path : {:<16}| Duration : {:<10} | Packet loss : {:.3f} | Weight : {:.3f}'.format(str(path_list[i]),str(delay_float_list[i]),packet_loss[i],float(avg_delay[i])))
            print('path=',path_list[i],'weight=',float(avg_delay[i]), file=f)


