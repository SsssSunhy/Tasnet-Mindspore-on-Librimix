# 目录

<!-- TOC -->

- [目录](#目录)
- [TasNet介绍](#TasNet介绍)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [数据预处理过程](#数据预处理过程)
        - [数据预处理](#数据预处理)
    - [训练过程](#训练过程)
        - [训练](#训练)  
    - [评估过程](#评估过程)
        - [评估](#评估)
    - [导出mindir模型](#导出mindir模型)
        - [导出](#导出)
    - [推理过程](#推理过程)
        - [推理](#推理)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

# TasNet介绍

TasNet使用编码器-解码器框架直接在时域中对信号进行建模，并对非负编码器输出执行源分离。该方法去除了频率分解步骤，并将分离问题简化为编码器输出上的源掩码估计，然后由解码器合成。该系统降低了语音分离的计算成本，并显着降低了输出所需的最小延迟。TasNet 适用于需要低功耗、实时实现的应用，例如可听设备和电信设备。

[论文](https://arxiv.org/pdf/1711.00541.pdf): TASNET: TIME-DOMAIN AUDIO SEPARATION NETWORK FOR REAL-TIME, SINGLE-CHANNEL SPEECH SEPARATION

# 模型架构

encoder：提取语音特征
separation：将encoder得到的结果传入一个4层的LSTM并进行分离
decoder：将分离结果进行处理，得到语音波形

# 数据集

使用的数据集为: [librimix](<https://catalog.ldc.upenn.edu/docs/LDC93S1/TIMIT.html>)，LibriMix 是一个开源数据集，用于在嘈杂环境中进行源代码分离。
要生成 LibriMix，请参照开源项目：https://github.com/JorisCos/LibriMix

# 环境要求

- 硬件（ASCEND）
    - ASCEND处理器
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 通过下面网址可以获得更多信息:
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)
- 依赖
    - 见requirements.txt文件，使用方法如下:

```python
pip install -r requirements.txt
```

# 脚本说明

## 脚本及样例代码

```path
TasNet
├── ascend310_infer
  ├─ build.sh                         # launch main.cc
  ├─ CMakeLists.txt                   # CMakeLists
  ├─ main.cc                          # 310 main function
├─ requirements.txt                   # requirements
├─ README.md                          # descriptions
├── scripts
  ├─ run_distribute_train.sh          # launch ascend training(8 pcs)
  ├─ run_stranalone_train.sh          # launch ascend training(1 pcs)
  ├─ run_eval.sh                      # launch ascend eval
  ├─ run_infer_310.sh                 # launch infer 310
├─ train.py                           # train script
├─ eval.py                            # eval
├─ preprocess.py                      # preprocess json
├─ data.py                            # postprocess data
├─ export.py                          # export mindir script
├─ network_define.py                  # define network
├─ tasnet.py                          # tasnet
├─ Loss.py                            # loss function
├─ train_wrapper.py                   # clip norm function
├─ preprocess_310.py                  # preprocess of 310
├─ postprocess.py                     # postprocess of 310

```

## 脚本参数

数据预处理、训练、评估的相关参数在`train.py`等文件

```text
数据预处理相关参数
in_dir                    预处理前加载原始数据集目录
out_dir                   预处理后的json文件的目录
sample_rate               采样率
```

```text
训练和模型相关参数
in_dir                     预处理前加载原始数据集目录
out_dir                    预处理后的json文件的目录
train_dir                  训练集
sample_rate                采样率
data_url                   云上训练数据路径
train_url                  云上训练模型存储位置
L                          语音分段每段长度
N                          基信号数量
hidden_size                LSTM隐藏层数量
num_layers                 LSTM层数
bidirectional              是否为双向LSTM
nspk                       说话人的数量
```

```text
评估相关参数
model_path                 ckpt路径
cal_sdr                    是否计算SDR
data_dir                   测试集路径
```

```text
配置相关参数
device_target              硬件，只支持ASCEND
device_id                  设备号
```

# 数据预处理过程

## 数据预处理

数据预处理运行示例:

```text
python preprocess.py
```

数据预处理过程很快，大约需要几分钟时间

# 训练过程

## 训练

- ### 单卡训练

运行示例:

```text
python train.py
```

或者可以运行脚本:

```bash
bash ./scripts/run_standalone_train.sh [DEVICE_ID]
```

可以通过train.log查看结果

- ### 分布式训练

分布式训练脚本如下

```bash
bash run_distribute_train.sh [DEVICE_NUM] [RANK_TABLE_FILE]
```

可以通过对应卡号的文件夹中的paralletrain.log查看结果

# 评估过程

## 评估

运行示例:

```text
python eval.py
参数:
model_path                 ckpt文件
data_dir                   测试集路径
```

或者可以运行脚本:

```bash
bash run_eval.sh [DEVICE_ID]
```

可以通过eval.log查看结果

# 导出mindir模型

## 导出

```bash
python export.py
```

# 推理过程

## 推理

### 用法

```bash
./scripts/run_infer_310.sh [MINDIR_PATH] [TEST_PATH] [NEED_PREPROCESS]
```

### 结果

```text
Average SISNR improvement: 5.97
```

# 模型描述

## 性能

### 训练性能

| 参数          | TasNet                          |
| ------------- | ------------------------------- |
| 资源          | Ascend910                       |
| 上传日期      | 2022-9-2                       |
| MindSpore版本 | 1.6.1                           |
| 数据集        | Librimix                        |
| 训练参数      | 8p, epoch = 50, batch_size = 4 |
| 优化器        | Adam                            |
| 损失函数      | SI-SNR                          |
| 输出          | SI-SNR(5.97)                   |
| 损失值        | -9.52                          |
| 运行速度      | 8p 5444.8 ms/step             |
| 训练总时间    | 8p: 约36h                      |

# 随机情况说明

随机性主要来自下面两点:

- 参数初始化
- 轮换数据集

# ModelZoo主页

 [ModelZoo主页](https://gitee.com/mindspore/models).
