


# 基于MobileNetV2的狗和牛角包二分类任务
2022年11月26日**更新**

在此教程中，我们将通过[MindSpore/MobileNetV2](https://xihe.mindspore.cn/projects/MindSpore/MobileNetV2)项目，快速体验图像二分类（狗和牛角包）任务的在线训练、推理。


# 目录  
[基本介绍](#基本介绍)  
- [任务简介](#任务简介)
- [项目地址](#项目地址)
- [项目结构](#项目结构)

[效果展示](#效果展示)
- [训练](#训练)
- [推理](#推理)

[快速开始](#快速开始)
- [Fork样例仓](#Fork样例仓)
- [在线推理](#在线推理)

[问题反馈](#问题反馈)



***
<a name="基本介绍"></a>
## 基本介绍

<a name="任务简介"></a>
### 任务简介

基于公开的模型仓库 MindSpore/MobileNetV2 进行模型训练，并使用仓库下的模型文件实现在线图像二分类推理。

#### MobileNetV2模型简介
MobileNetV1是为移动和嵌入式设备提出的轻量级模型。MobileNets使用深度可分离卷积来构建轻量级深度神经网络，进而实现模型在算力有限的情况下的应用。
MobileNet是基于深度可分离卷积的。深度可分离卷积把标准卷积分解成深度卷积(depthwise convolution)和逐点卷积(pointwise convolution)，进而大幅度降低参数量和计算量。深度可分离卷积示意图如下：
<img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/mobilenetv2/mobilenet_deep_conv.PNG" width="70%">

深度可分离卷积将标准的卷积操作(a)拆分为：(b)深度卷积和(c)逐点卷积，从而减少计算量。

MobileNetV2是在MobileNetV1的基础上提出一种新型层结构： 具有线性瓶颈的倒残差结构(the inverted residual with linear bottleneck)。模型结构如下：

<img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/mobilenetv2/mobilenetv2_model.PNG" width="80%">

MobileNetv2在极少的参数量下有着不错的性能。

#### 相关论文
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html)

    
#### 数据集简介
使用的数据集包括两个类别的图片：**狗和牛角包**，数据集结构如下：
```
 ├── train    # 训练集
 │  ├── croissants   # 牛角包图片
 │  ├── dog   # 狗图片
 │── val   # 验证集
 │  ├── croissants   # 牛角包图片
 │  ├── dog   # 狗图片
```

<a name="项目地址"></a>
### 项目地址
- 模型仓库：[MindSpore/MobileNetV2](https://xihe.mindspore.cn/projects/MindSpore/MobileNetV2)
- 数据集地址：[drizzlezyk/MobileNetV2_image](https://xihe.mindspore.cn/datasets/drizzlezyk/MobileNetV2_image)
- 模型地址：[MindSpore/MobileNetV2_model](https://xihe.mindspore.cn/models/MindSpore/MobileNetV2_model)


<a name="项目结构"></a>
### 项目结构

项目的目录分为两个部分：推理（inference）和训练（train），推理可视化相关的代码放在inference文件夹下，训练相关的代码放在train文件夹下。

```python
 ├── inference    # 推理可视化相关代码目录
 │  ├── app.py    # 推理核心启动文件
 │  ├── requirements.txt    # 推理可视化相关依赖文件
 │  ├── config.json
 └── train    # 在线训练相关代码目录
   └── train_dir         # 训练代码所在的目录
     ├── pip-requirements.txt  # 训练代码所需要的package依赖声明文件
     └── train.py       # 神经网络训练代码
```


***
<a name="效果展示"></a>
## 效果展示

<a name="训练"></a>
### 训练

 <img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/resnet50/trainlog.PNG" width="80%">

<a name="推理"></a>
### 推理
<img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/mobilenetv2/mobilenetv2_inference.PNG" width="80%">

***
<a name="快速开始"></a>
## 快速开始

<a name="Fork样例仓"></a>
### Fork样例仓

1. 在项目搜索框中输入MobileNetV2，找到样例仓 **MindSpore/MobileNetV2**

2. 点击Fork

<a name="在线训练"></a>
### 在线训练

创建训练后，就可以通过普通日志和可视化日志观察训练动态。

1. 选择“**训练**”页签，点击“**创建训练实例**”，在线填写表单，首先填写训练名称，选择对应的代码目录、启动文件。

   <img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/mobilenetv2/train_form1.PNG" width="70%">

2. 输入模型、数据集、输出路径等超参数指定：
- 在表单中指定使用的预训练模型文件存放路径（文件存放在昇思大模型平台的模型模块下）
- 在表单中指定使用的数据集文件存放路径（文件存放在昇思大模型平台的数据集模块下）
- 训练的输出结果统一指定超参数名：output_path，需要在代码的argparse模块声明

   <img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/mobilenetv2/train_form2.PNG" width="70%">

    
3. 点击创建训练，注意一个仓库同时只能有一个运行中的训练实例，且训练实例最多只能5个。

4. 查看训练列表：将鼠标放置于“**训练**”栏上，点击训练下拉框中的“**训练列表**”即可。
  
   <img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/resnet50/resnet-train-list.PNG" width="70%">

 

5. 查看训练日志：点击训练名称，即可进入该训练的详情页面：

   <img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/resnet50/train_info.PNG" width="70%">

- 所有输出到超参数output_path的文件都在tar.gz文件中。

<a name="在线推理"></a>
### 在线推理

本项目的推理模块是将训练好的模型应用到实时的图像风格迁移任务中，可将一些生活中自然风光的照片和艺术家的画作进行相互的风格迁移，具体如下：


1. 选择“**推理**”页签，点击“**启动**”按钮。

2. 等待2分钟左右，出现推理可视化界面，提交一张狗或者牛角包的图片，点击submit进行预测：

     <img src="https://obs-xihe-beijing4.obs.cn-north-4.myhuaweicloud.com/xihe-img/projects/quick_start/mobilenetv2/mobilenetv2_inference.PNG" width="70%">

***
<a name="问题反馈"></a>
# 问题反馈

本教程会持续更新，您如果按照教程在操作过程中出现任何问题，请您随时在我们的[官网仓](https://gitee.com/mindspore/mindspore)提ISSUE，我们会及时回复您。如果您有任何建议，也可以添加官方助手小猫子（微信号：mindspore0328），我们非常欢迎您的宝贵建议，如被采纳，会收到MindSpore官方精美礼品哦！
