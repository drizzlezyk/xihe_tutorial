import argparse

from mindvision.classification.dataset import Mnist
from mindvision.classification.models import lenet

import mindspore.nn as nn
from mindspore import load_checkpoint, load_param_into_net
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.callback import Callback, LossMonitor

from aim import Run
import moxing as mox


def parse_args():
    # 创建解析
    parser = argparse.ArgumentParser(description="train mnist",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--pretrain_url', type=str, default='', help='pretrain model path')
    parser.add_argument('--data_url', type=str, default='./mnist', help='training data path')
    # 输出路径，输出的权重文件或者图片需要指定在此路径下
    parser.add_argument('--output_path', default='', type=str, help='model saved path')
    # 自定义评估时需要用到此参数
    parser.add_argument('--aim_repo', default='', type=str, help='aim repo saved path')
    # 解析参数
    args_opt = parser.parse_args()
    return args_opt



# 自定义Callback
class AimCallback(Callback):
    def __init__(self, model, dataset_test, aim_run):
        super(AimCallback, self).__init__()
        self.aim_run = aim_run  # 传入aim实例
        self.model = model  # 传入model，用于eval
        self.dataset_test = dataset_test  # 传入dataset_test, 用于eval test

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        # loss
        epoch_num = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num
        loss = cb_params.net_outputs
        run1 = self.aim_run.track(float(str(loss)), name='loss', step=step_num, epoch=epoch_num,
                                  context={"subset": "train"})

    def epoch_end(self, run_context):
        """epoch end"""
        cb_params = run_context.original_args()
        # loss
        epoch_num = cb_params.cur_epoch_num
        step_num = cb_params.cur_step_num
        loss = cb_params.net_outputs
        train_dataset = cb_params.train_dataset
        train_acc = self.model.eval(train_dataset)
        test_acc = self.model.eval(self.dataset_test)
        print("【Epoch:】", epoch_num, "【Step:】", step_num, "【loss:】", loss, "【train_acc:】", train_acc['accuracy'], "【test_acc:】",
              test_acc['accuracy'])

        self.aim_run.track(float(str(loss)), name='loss', epoch=epoch_num, context={"subset": "train"})
        self.aim_run.track(float(str(train_acc['accuracy'])), name='accuracy', epoch=epoch_num,
                           context={"subset": "train"})
        self.aim_run.track(float(str(test_acc['accuracy'])), name='accuracy', epoch=epoch_num,
                           context={"subset": "test"})


def train(args_opt):
    # ablation experiments
    batch_size_choice = [32, 64, 128]
    learning_rate_choice = [0.01, 0.001, 0.0001]
    momentum_choice = [0.9, 0.99]

    for batch_size in batch_size_choice:
        for learning_rate in learning_rate_choice:
            for momentum in momentum_choice:
                # 将模型参数存入parameter的字典中，这里加载的是上面训练过程中保存的模型参数
                param_dict = load_checkpoint(args_opt.pretrain_url)
                # 重新定义一个LeNet神经网络
                network = lenet(num_classes=10, pretrained=False)
                # 将参数加载到网络中
                load_param_into_net(network, param_dict)

                # 定义损失函数
                net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
                # 定义优化器函数
                net_opt = nn.Momentum(network.trainable_params(), learning_rate=learning_rate, momentum=momentum)
                model = Model(network, loss_fn=net_loss, optimizer=net_opt, metrics={"accuracy"})

                # 定义训练数据集
                download_train = Mnist(path=args_opt.data_url, split="train", batch_size=batch_size, repeat_num=1, shuffle=True, resize=32,
                                       download=True)
                # 定义测试数据集
                download_eval = Mnist(path=args_opt.data_url, split="test", batch_size=batch_size, repeat_num=1, shuffle=True, resize=32,
                                      download=True)
                dataset_train = download_train.run()
                dataset_test = download_eval.run()

                # 设置模型保存参数
                config_ck = CheckpointConfig(save_checkpoint_steps=int(60000/batch_size), keep_checkpoint_max=20)

                # 应用模型保存参数
                ckpoint = ModelCheckpoint(prefix="lenet", directory=args_opt.output_path, config=config_ck)
                # Aim
                aim_run = Run(repo=args_opt.aim_repo, experiment=f"{args_opt.output_path}/bs{batch_size}_lr{learning_rate}_mt{momentum}")

                # Log run parameters
                aim_run['learning_rate'] = learning_rate
                aim_run['momentum'] = momentum
                aim_run['batch_size'] = batch_size
                
                # 训练网络模型
                model.train(5, dataset_train, callbacks=[ckpoint, AimCallback(model, dataset_test, aim_run)])


if __name__ == '__main__':
    args_opt = parse_args()
    train(args_opt)
