# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import argparse
from typing import Type, Union, List, Optional

from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net, nn

from mindvision.classification.models.blocks import ConvNormActivation
from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.head import DenseHead
from mindvision.classification.models.neck import GlobalAvgPooling
from mindvision.classification.utils.model_urls import model_urls
from mindvision.classification.dataset import Cifar10
from mindvision.utils.load_pretrained_model import LoadPretrainedModel
from mindvision.engine.callback import ValAccMonitor


def parse_args():
    """
    对命令行参数进行解析
    :return: parser.parse_args()
    """
    # 创建解析
    parser = argparse.ArgumentParser(description="train resnet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--pretrain_path',
                        type=str,
                        default='models/drizzlezyk/resnet50_model/resnet50_224.ckpt',
                        help='the pretrain file')

    parser.add_argument('--data_path',
                        type=str,
                        default='datasets/drizzlezyk/cifar10/',
                        help='the training data')

    parser.add_argument('--output_path',
                        default='train/resnet/',
                        type=str,
                        help='the path model saved')

    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help='training epochs')

    parser.add_argument('--lr',
                        default=0.0001,
                        type=int,
                        help='training epochs')

    return parser.parse_args()


# -------------------------------- building block -------------------------------------------
class ResidualBlockBase(nn.Cell):
    """残差模块"""
    expansion: int = 1  # 最后一个卷积核数量与第一个卷积核数量相等

    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None) -> None:
        super().__init__()
        if not norm:
            norm = nn.BatchNorm2d

        self.conv1 = ConvNormActivation(in_channel, out_channel,
                                        kernel_size=3, stride=stride, norm=norm)
        self.conv2 = ConvNormActivation(out_channel, out_channel,
                                        kernel_size=3, norm=norm, activation=None)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        """ResidualBlockBase construct."""
        identity = x  # shortcuts分支

        out = self.conv1(x)  # 主分支第一层：3*3卷积层
        out = self.conv2(out)  # 主分支第二层：3*3卷积层

        if self.down_sample:
            identity = self.down_sample(x)
        out += identity  # 输出为主分支与shortcuts之和
        out = self.relu(out)

        return out


# --------------------------- Bottleneck --------------------------------------
class ResidualBlock(nn.Cell):
    expansion = 4  # 最后一个卷积核的数量是第一个卷积核数量的4倍

    def __init__(self, in_channel: int, out_channel: int,
                 stride: int = 1, norm: Optional[nn.Cell] = None,
                 down_sample: Optional[nn.Cell] = None) -> None:
        super(ResidualBlock, self).__init__()
        if not norm:
            norm = nn.BatchNorm2d

        self.conv1 = ConvNormActivation(in_channel,
                                        out_channel,
                                        kernel_size=1,
                                        norm=norm)
        self.conv2 = ConvNormActivation(out_channel,
                                        out_channel,
                                        kernel_size=3,
                                        stride=stride,
                                        norm=norm)
        self.conv3 = ConvNormActivation(out_channel,
                                        out_channel * self.expansion,
                                        kernel_size=1,
                                        norm=norm,
                                        activation=None)
        self.relu = nn.ReLU()
        self.down_sample = down_sample

    def construct(self, x):
        identity = x  # shortscuts分支

        out = self.conv1(x)  # 主分支第一层：1*1卷积层
        out = self.conv2(out)  # 主分支第二层：3*3卷积层
        out = self.conv3(out)  # 主分支第三层：1*1卷积层

        if self.down_sample:
            identity = self.down_sample(x)

        out += identity  # 输出为主分支与shortcuts之和
        out = self.relu(out)

        return out


def make_layer(last_out_channel, block: Type[Union[ResidualBlockBase, ResidualBlock]],
               channel: int, block_nums: int, stride: int = 1):
    down_sample = None  # shortcuts分支

    if stride != 1 or last_out_channel != channel * block.expansion:
        down_sample = ConvNormActivation(last_out_channel,
                                         channel * block.expansion,
                                         kernel_size=1,
                                         stride=stride,
                                         norm=nn.BatchNorm2d,
                                         activation=None)
    layers = []
    layers.append(block(last_out_channel,
                        channel,
                        stride=stride,
                        down_sample=down_sample,
                        norm=nn.BatchNorm2d))

    in_channel = channel * block.expansion
    # 堆叠残差网络
    for _ in range(1, block_nums):
        layers.append(block(in_channel, channel, norm=nn.BatchNorm2d))

    return nn.SequentialCell(layers)


class ResNet(nn.Cell):
    """ResNet网络"""
    def __init__(self, block: Type[Union[ResidualBlockBase, ResidualBlock]],
                 layer_nums: List[int], norm: Optional[nn.Cell] = None) -> None:
        super().__init__()
        if not norm:
            norm = nn.BatchNorm2d
        # 第一个卷积层，输入channel为3（彩色图像），输出channel为64
        self.conv1 = ConvNormActivation(3, 64, kernel_size=7, stride=2, norm=norm)
        # 最大池化层，缩小图片的尺寸
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        # 各个残差网络结构块定义，
        self.layer1 = make_layer(64, block, 64, layer_nums[0])
        self.layer2 = make_layer(64 * block.expansion, block, 128, layer_nums[1], stride=2)
        self.layer3 = make_layer(128 * block.expansion, block, 256, layer_nums[2], stride=2)
        self.layer4 = make_layer(256 * block.expansion, block, 512, layer_nums[3], stride=2)

    def construct(self, x):
        """构建ResNet"""
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def _resnet(arch: str, block: Type[Union[ResidualBlockBase, ResidualBlock]],
            layers: List[int], num_classes: int, pretrained: bool, input_channel: int):
    backbone = ResNet(block, layers)
    neck = GlobalAvgPooling()  # 平均池化层
    head = DenseHead(input_channel=input_channel, num_classes=num_classes)  # 全连接层
    model = BaseClassifier(backbone, neck, head)  # 将backbone层、neck层和head层连接起来

    if pretrained:
        # 下载并加载预训练模型
        LoadPretrainedModel(model, model_urls[arch]).run()

    return model


def resnet50(num_classes: int = 1000, pretrained: bool = False):
    "ResNet50模型"
    return _resnet("resnet50", ResidualBlock, [3, 4, 6, 3], num_classes, pretrained, 2048)


def train(args_opt):
    """
    加载数据集，训练ResNet50
    :param args_opt: 命令行传入的超参数
    :return: None
    """
    # 加载CIFAR-10训练数据集
    dataset_train = Cifar10(path=args_opt.data_path,
                            split='train',
                            batch_size=128,
                            resize=32,
                            download=False)
    ds_train = dataset_train.run()
    step_size = ds_train.get_dataset_size()

    # 加载CIFAR-10测试数据集
    dataset_val = Cifar10(path=args_opt.data_path,
                          split='test',
                          batch_size=128,
                          resize=32,
                          download=False)
    ds_val = dataset_val.run()
    data = next(ds_train.create_dict_iterator())
    print("Image shape:", data["image"].asnumpy().shape, ", Label:", data["label"].asnumpy())

    # 定义ResNet50网络
    network = resnet50(pretrained=False)
    param_dict = load_checkpoint(args_opt.pretrain_path)
    load_param_into_net(network, param_dict)

    # 全连接层输入层的大小
    in_channel = network.head.dense.in_channels
    # 重置全连接层
    network.head = DenseHead(input_channel=in_channel, num_classes=10)

    learning_rate = nn.cosine_decay_lr(min_lr=0.00001,
                                       max_lr=0.001,
                                       total_step=step_size * args_opt.epochs,
                                       step_per_epoch=step_size,
                                       decay_epoch=args_opt.epochs)

    optimizer = nn.Momentum(params=network.trainable_params(),
                            learning_rate=learning_rate,
                            momentum=0.9)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True,
                                            reduction='mean')

    model = Model(network,
                  loss,
                  optimizer,
                  metrics={"Accuracy": nn.Accuracy()})

    print('===Start training===')
    model.train(args_opt.epochs,
                ds_train,
                callbacks=[ValAccMonitor(model,
                                         ds_val,
                                         args_opt.epochs,
                                         ckpt_directory=args_opt.output_path)])
    print('===Finish training===')


if __name__ == '__main__':
    train(parse_args())
