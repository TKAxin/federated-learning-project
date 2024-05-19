## 参数文件utils/conf.json

修改参数设置

* model_name：模型名称
* no_models：客户端数量
* type：数据集信息
* global_epochs：全局迭代次数，即服务端与客户端的通信迭代次数
* local_epochs：本地模型训练迭代次数
* k：每一轮迭代时，服务端会从所有客户端中挑选k个客户端参与训练。
* batch_size：本地训练每一轮的样本数
* lr，momentum，lambda：本地训练的超参数设置
