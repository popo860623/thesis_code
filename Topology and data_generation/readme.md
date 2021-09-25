# Flie description

* metric_QPSP_priority_generator:
	Generate opt_path file used for metric & QPSP algo

* mininet:
	Program used to create NSFNet topology & GBNNet topology

* NetworkX_graph_file:
	read by mininet program (topology info)

* TFRecord transformer:
	Transform ping log file to tfrecord which is used for model training

	
	
# 資料產生流程:
主要分為三部分:
* Mininet 建立拓樸、產生資料集
* SDN Controller + 各演算法
* 將資料集轉換為TFRecord，用於訓練 RouteNet



1. 建立拓樸(以NSFNet為例)
```=shell
$ cd ~/thesis_code/Topology_and_data_generation/mininet/
$ sudo python nsfnet_topo.py
```

2. 啟動SDN Controller
```=shell
$ cd ~/thesis_code/controller/
$ ryu-manager --observe-links main.py
```
Line.4 ~ Line.13為各種演算法之Handler，根據使用的演算法反註解就好
```python3=
class ShortestPath(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {
        # "ArpHandler": ArpHandler_opt.ArpHandler
        # "ArpHandler": ArpHandler_opt_gbn.ArpHandler
        # "ArpHandler": ArpHandler_ML3.ArpHandler
        # "ArpHandler": ArpHandler_ML_gbn.ArpHandler
         "ArpHandler": ArpHandler_SP.ArpHandler
        # "ArpHandler": ArpHandler_SP_gbn.ArpHandler
        # "ArpHandler": ArpHandler_RPSP.ArpHandler
        # "ArpHandler": ArpHandler_RPSP_gbn.ArpHandler
        # "ArpHandler": ArpHandler_QPSP.ArpHandler
        # "ArpHandler": ArpHandler_QPSP_gbn.ArpHandler
    }
```

3. Mininet啟動拓樸後，會看到以下畫面 (e.g. nsfnet_topo.py):
     {%youtube AiJXgMgVckc %}


    有**三個指令**可以控制

    * GEN : 用於產生Flow data set(產生資料集與實驗Utilization都會用到)
    {%youtube 7gsySbcjqJI %}
    * 此處可以配置寫入路徑
    ![](https://i.imgur.com/JX3HK96.png)

    * CLI : 進入原Mininet command line interface(CLI介面) (可下Pingall等Mininet提供的指令)
    {%youtube wLS8K7QmcJk %}

    * QUIT : 關閉Mininet

    
4. 若資料已產生完畢，要使用這些資料訓練模型時，要將其轉為TFRecord進行訓練(data_generator_nsf.py):
5. 
   1. 輸入由ping log目錄
   ```python=22
   traffic_dir_op='example/traffic_dir/'
   ```

   2. 在 make_tfrecord function底下，writer為TFRecord寫入路徑:
   ```python=103
   if not os.path.exists('/home/hao/thesis_code/example/tfrecords/'):
        os.makedirs('/home/hao/thesis_code/example/tfrecords/')
    writer = tf.io.TFRecordWriter('/home/hao/thesis_code/example/tfrecords/' + str(i) + '.tfrecords')
   ```
   
   3. 切割資料集(Train、validation、test)
       * Line 143~145為三個切割後資料集儲存路徑
       * Line153為切割之前的資料集路徑(第二點writer的路徑)
   ```python=141
   def split_tfrecords():
    print('********************************************')
    tfr_train = '/home/hao/thesis_code/example/tfrecords/' + 'train_example/'
    tfr_evaluate = '/home/hao/thesis_code/example/tfrecords/' + 'eval_example/'
    tfr_test = '/home/hao/thesis_code/example/tfrecords/' + 'test_example/'
    if not os.path.exists(tfr_train):
        os.makedirs(tfr_train)
    if not os.path.exists(tfr_evaluate):
        os.makedirs(tfr_evaluate)
    if not os.path.exists(tfr_test):
        os.makedirs(tfr_test)

    tfrecords = glob.glob('/home/hao/thesis_code/example/tfrecords/' + '*.tfrecords')
   ```
   

    4. 執行
     ```=shell
   sudo python3 TFRecord transformer/data_generator_nsf.py
   ```
    {%youtube cYpice0uS8Q %}
    
#### 訓練
將資料都轉為TFRecord後，可開始訓練，參考:
[Train RouteNet Model](https://hackmd.io/1uN64n13Rimi45pzZGAqWQ?view)

