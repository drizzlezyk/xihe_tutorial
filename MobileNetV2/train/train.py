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

import mindspore as ms
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as vision

from mindspore import nn
from mindspore.train.callback import TimeMonitor
from mindvision.classification.models import mobilenet_v2
from mindvision.engine.loss import CrossEntropySmooth
from mindvision.engine.callback import ValAccMonitor


def parse_args():
    """
    解析参数
    :return: parser.parse_args()
    """
    # 创建解析
    parser = argparse.ArgumentParser(description="train resnet",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--pretrain_path',
                        type=str,
                        default='datasets/drizzlezyk/cifar10/resnet50_224.ckpt',
                        help='the pretrain file')

    parser.add_argument('--data_path',
                        type=str,
                        default='datasets/drizzlezyk/cifar10/',
                        help='the training data')

    parser.add_argument('--output_path',
                        default='train/output/',
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


def load_dataset(path, batch_size=64, is_train=True, image_size=224):
    dataset = ds.ImageFolderDataset(path,
                                    num_parallel_workers=8,
                                    class_indexing={"croissants": 0, "dog": 1})

    # 图像增强操作
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    if is_train:
        trans = [
            vision.RandomCropDecodeResize(image_size,
                                          scale=(0.08, 1.0),
                                          ratio=(0.75, 1.333)),
            vision.RandomHorizontalFlip(prob=0.5),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]
    else:
        trans = [
            vision.Decode(),
            vision.Resize(256),
            vision.CenterCrop(image_size),
            vision.Normalize(mean=mean, std=std),
            vision.HWC2CHW()
        ]

    dataset = dataset.map(operations=trans,
                          input_columns="image",
                          num_parallel_workers=8)
    # 设置batch_size的大小，若最后一次抓取的样本数小于batch_size，则丢弃
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def train(args_opt):
    """
    加载数据集，训练MobileNetV2
    :param args_opt: 命令行参数
    :return: None
    """
    # 加载数据集
    dataset_train = load_dataset(args_opt.data_path+"/train",
                                 is_train=True)
    dataset_val = load_dataset(args_opt.data_path+"/val",
                               is_train=False,
                               batch_size=2)

    # 创建模型,其中目标分类数为2，图像输入大小为(224,224)
    network = mobilenet_v2(num_classes=2, resize=224)

    # 模型参数存入到param_dict
    param_dict = ms.load_checkpoint(args_opt.pretrain_path)

    # 获取mobilenet_v2网络最后一个卷积层的参数名
    filter_list = [x.name for x in network.head.classifier.get_parameters()]

    # 删除预训练模型的最后一个卷积层
    def filter_ckpt_parameter(origin_dict, param_filter):
        for key in list(origin_dict.keys()):
            for name in param_filter:
                if name in key:
                    print("Delete parameter from checkpoint: ", key)
                    del origin_dict[key]
                    break

    filter_ckpt_parameter(param_dict, filter_list)

    # 加载预训练模型参数作为网络初始化权重
    ms.load_param_into_net(network, param_dict)

    # 定义优化器
    network_opt = nn.Momentum(params=network.trainable_params(),
                              learning_rate=0.01,
                              momentum=0.9)

    # 定义损失函数
    network_loss = CrossEntropySmooth(sparse=True,
                                      reduction="mean",
                                      smooth_factor=0.1,
                                      classes_num=2)

    # 定义评价指标
    metrics = {"Accuracy": nn.Accuracy()}

    # 初始化模型
    model = ms.Model(network,
                     loss_fn=network_loss,
                     optimizer=network_opt,
                     metrics=metrics)

    # 模型训练与验证，训练完成后保存验证精度最高的ckpt文件（best.ckpt）到当前目录下
    model.train(args_opt.epochs,
                dataset_train,
                callbacks=[ValAccMonitor(model,
                                         dataset_val,
                                         args_opt.epochs,
                                         ckpt_directory=args_opt.output_path),
                           TimeMonitor()])


if __name__ == '__main__':
    train(parse_args())
