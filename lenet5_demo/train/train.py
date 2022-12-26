import argparse
import os

from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet

import mindspore
import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindvision.engine.callback import LossMonitor


def parse_args():
    # 创建解析
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--pretrain_url', type=str, default='', help='pretrain model path')
    parser.add_argument('--data_url', type=str, default='', help='training data path')
    # 输出路径，输出的权重文件或者图片需要指定在此路径下
    parser.add_argument('--output_path', default='', type=str, help='model saved path')
    # 解析参数
    args_opt = parser.parse_args()
    return args_opt



def train(args_opt):
    # 将模型参数存入parameter的字典中，这里加载的是上面训练过程中保存的模型参数
    param_dict = load_checkpoint(args_opt.pretrain_url)

    # 重新定义一个LeNet神经网络
    network = lenet(num_classes=10, pretrained=False)

    # 将参数加载到网络中
    load_param_into_net(network, param_dict)
    # 定义损失函数
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # 定义优化器函数
    net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
    model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={"accuracy"})

    # 定义训练数据集
    download_train = Mnist(path=args_opt.data_url, split="train", batch_size=32, repeat_num=1, shuffle=True, resize=32,
                           download=True)
    dataset_train = download_train.run()

    # 设置模型保存参数
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=1)

    # 应用模型保存参数
    ckpoint = ModelCheckpoint(prefix="lenet", directory=args_opt.output_path, config=config_ck)

    # 训练网络模型
    model.train(10, dataset_train, callbacks=[ckpoint, LossMonitor(0.01, 1875)])


if __name__ == '__main__':
    print(mindspore.run_check())
    os.system("pwd")
    os.system("ls -lrt")
    args_opt = parse_args()
    train(args_opt)
