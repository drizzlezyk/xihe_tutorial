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
# ============================================================================

""" LeNet training add Aim and grid search """


import argparse

from mindspore import load_checkpoint, load_param_into_net, nn
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindvision.engine.callback import LossMonitor
from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet


def parse_args():
    # 创建解析
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--pretrain_url',
                        type=str, default='',
                        help='pretrain model path')

    parser.add_argument('--data_url',
                        type=str,
                        default='',
                        help='training data path')

    parser.add_argument('--output_path',
                        default='',
                        type=str,
                        help='model saved path')
    # 解析参数
    return parser.parse_args()


def train():
    batch_size_choice = [32, 64, 128]
    learning_rate_choice = [0.01, 0.001, 0.0001]
    momentum_choice = [0.9, 0.99]

    for batch_size in batch_size_choice:
        for learning_rate in learning_rate_choice:
            for momentum in momentum_choice:
                # 将模型参数存入parameter的字典中，这里加载的是上面训练过程中保存的模型参数
                param_dict = load_checkpoint(args_opt.pretrain_url)
                # 初始化一个LeNet神经网络
                network = lenet(num_classes=10, pretrained=False)
                # 将参数加载到网络中
                load_param_into_net(network, param_dict)

                # 定义损失函数
                net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True,
                                                            reduction='mean')
                # 定义优化器函数
                net_opt = nn.Momentum(network.trainable_params(),
                                      learning_rate=learning_rate,
                                      momentum=momentum)
                model = Model(network, loss_fn=net_loss,
                              optimizer=net_opt, metrics={"accuracy"})

                # 定义训练数据集
                dataset_train = Mnist(path=args_opt.data_url,
                                      split="train",
                                      batch_size=batch_size,
                                      repeat_num=1,
                                      shuffle=True,
                                      resize=32,
                                      download=True).run()

                steps = int(60000/batch_size)
                # 设置模型保存参数
                config_checkpoint = CheckpointConfig(save_checkpoint_steps=steps,
                                                     keep_checkpoint_max=20)
                print(f"bs{batch_size}_lr{learning_rate}_mt{momentum}/{args_opt.output_path}")
                # 应用模型保存参数
                checkpoint = ModelCheckpoint(prefix="lenet",
                                             directory=f"{args_opt.output_path}/bs{batch_size}"
                                                       f"_lr{learning_rate}_mt{momentum}",
                                             config=config_checkpoint)
                # 训练网络模型
                model.train(20,
                            dataset_train,
                            callbacks=[checkpoint,
                                       LossMonitor(learning_rate, steps)])


if __name__ == '__main__':
    args_opt = parse_args()
    train()
