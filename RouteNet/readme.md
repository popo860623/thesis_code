# Train RouteNet Model
:::success
RouteNet訓練步驟說明
:::
1. 訓練模型前需要先準備資料集，分別是**Training set & Validation set** ([參考](https://github.com/popo860623/thesis_code/tree/main/Topology%20and%20data_generation))
2. 啟動訓練的檔案為 **run_routenet.sh**

### run_routenet.sh
以下介紹訓練模型時會使用到的參數:
* --hparams : 設定超參數
* --train : 設定training set的路徑
* --train_step : 訓練步數
* -- eval_ : 設定validation set的路徑
* --model_dir : 模型存放路徑

```shell=
if [[ "$1" = "train" ]]; then

    python3 routenet_with_link_cap2.py train \
        --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=64,path_state_dim=64,$
        --train  $2/train_example/*.tfrecords\
        --train_steps $3\
        --eval_ $2/eval_example/*.tfrecords\ 
        --model_dir ./example_model/
fi

```
### Run training
```shell=
$ cd /home/hao/RouteNet/
$ ./run_routenet.sh train /home/hao/example/tfrecords/ 100000
```

![](https://github.com/popo860623/thesis_code/blob/main/doc/training.gif?raw=true)

### Evaluation
模型訓練完成後，要測試模型的效果請參考:
[Demo RouteNet](https://hackmd.io/SvDHWK8zQmagns5XAhT6DQ?view)
