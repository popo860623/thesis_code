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



1. 建立拓樸
```=shell
$sudo python <topo_dir>/<topo_file_name>.py
```

2. 啟動SDN Controller
```=shell
$ryu-manager --observe-links main.py <controller_dir>/main.py
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
        # "ArpHandler": ArpHandler_SP.ArpHandler
        # "ArpHandler": ArpHandler_SP_gbn.ArpHandler
        "ArpHandler": ArpHandler_RPSP.ArpHandler
        # "ArpHandler": ArpHandler_RPSP_gbn.ArpHandler
        # "ArpHandler": ArpHandler_QPSP.ArpHandler
        # "ArpHandler": ArpHandler_QPSP_gbn.ArpHandler
    }
```

3. Mininet啟動拓樸後，會看到以下畫面 (e.g. nsfnet_topo.py):

    ![](https://i.imgur.com/SNaOF87.png)

    * GEN : 用於產生Flow(產生資料集與實驗Utilization都會用到)
    * CLI : 進入Mininet command line interface (可下Pingall等Mininet提供的指令)
    * QUIT : 關閉Mininet

    
4. 若資料已產生完畢，要使用這些資料訓練模型時，要將其轉為TFRecord進行訓練:
   ```=shell
   sudo python3 <dir>/data_generator_<topo_name>.py
   ```
   1. 輸入由ping log目錄
   ![](https://i.imgur.com/HRAvIXD.png)
   2. 在 make_tfrecord function底下，writer為TFRecord寫入路徑:
   ![](https://i.imgur.com/0WJuqyB.png)
