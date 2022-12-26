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

""" DCGAN training 指定训练使用的平台为GPU，如需使用昇腾硬件可将其替换为Ascend """

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import mindspore
import mindspore.dataset as ds

from mindspore import context, nn, ops, ms_function
from mindspore.common.initializer import Normal
from mindspore.dataset import vision

from matplotlib import animation


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


def parse_args():
    # 创建解析
    parser = argparse.ArgumentParser(description="train dcgan",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--pretrain_url',
                        type=str,
                        default='',
                        help='the pretrain model path')

    parser.add_argument('--data_url',
                        type=str,
                        default='',
                        help='the training data path')

    parser.add_argument('--output_path',
                        default='',
                        type=str,
                        help='the path model saved')
    # 解析参数
    return parser.parse_args()


def create_dataset_imagenet(dataset_path, batch_size=128):
    """数据加载"""
    image_size = 64
    data_set = ds.ImageFolderDataset(dataset_path,
                                     num_parallel_workers=4,
                                     shuffle=True,
                                     decode=True)

    # 数据增强操作
    transforms = [
        vision.Resize(image_size),
        vision.CenterCrop(image_size),
        vision.HWC2CHW(),
        lambda x: ((x / 255).astype("float32"))
    ]

    # 数据映射操作
    data_set = data_set.project('image')
    data_set = data_set.map(transforms, 'image')

    # 批量操作
    data_set = data_set.batch(batch_size)
    return data_set


def preview_data(data, storage_path):
    data_iter = next(data.create_dict_iterator(output_numpy=True))
    # 可视化部分训练数据
    plt.figure(figsize=(10, 3), dpi=140)
    for i, image in enumerate(data_iter['image'][:30], 1):
        plt.subplot(3, 10, i)
        plt.axis("off")
        plt.imshow(image.transpose(1, 2, 0))
    plt.savefig(storage_path + "/preview.jpg")


class Generator(nn.Cell):
    """DCGAN网络生成器"""

    def __init__(self, ngf_layer=64, n_chanel=3):
        super().__init__()
        self.generator = nn.SequentialCell(
                nn.Conv2dTranspose(n_mid, ngf_layer * 8, 4, 1, 'valid', weight_init=weight_init),
                nn.BatchNorm2d(ngf_layer * 8, gamma_init=gamma_init),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf_layer * 8,
                                   ngf_layer * 4, 4, 2,
                                   'pad', 1,
                                   weight_init=weight_init),
                nn.BatchNorm2d(ngf_layer * 4, gamma_init=gamma_init),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf_layer * 4,
                                   ngf_layer * 2, 4, 2,
                                   'pad', 1,
                                   weight_init=weight_init),
                nn.BatchNorm2d(ngf_layer * 2, gamma_init=gamma_init),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf_layer * 2,
                                   ngf_layer, 4, 2,
                                   'pad', 1,
                                   weight_init=weight_init),
                nn.BatchNorm2d(ngf_layer, gamma_init=gamma_init),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf_layer,
                                   n_chanel, 4, 2,
                                   'pad', 1,
                                   weight_init=weight_init),
                nn.Tanh()
            )

    def construct(self, input_vector):
        return self.generator(input_vector)


class Discriminator(nn.Cell):
    """DCGAN网络判别器"""

    def __init__(self, ngf_layer=64, ndf_layer=64, n_chanel=3):
        super().__init__()
        self.discriminator = nn.SequentialCell(
                nn.Conv2d(n_chanel, ndf_layer, 4, 2, 'pad', 1, weight_init=weight_init),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ndf_layer, ndf_layer * 2, 4, 2, 'pad', 1, weight_init=weight_init),
                nn.BatchNorm2d(ngf_layer * 2, gamma_init=gamma_init),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ndf_layer * 2, ndf_layer * 4, 4, 2, 'pad', 1, weight_init=weight_init),
                nn.BatchNorm2d(ngf_layer * 4, gamma_init=gamma_init),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ndf_layer * 4, ndf_layer * 8, 4, 2, 'pad', 1, weight_init=weight_init),
                nn.BatchNorm2d(ngf_layer * 8, gamma_init=gamma_init),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ndf_layer * 8, 1, 4, 1, 'valid', weight_init=weight_init),
            )
        self.adv_layer = nn.Sigmoid()

    def construct(self, input_vector):
        out = self.discriminator(input_vector)
        out = out.reshape(out.shape[0], -1)
        return self.adv_layer(out)


def generator_forward(real_images, valid):
    # Sample noise as generator input
    vector_z = ops.StandardNormal()((real_images.shape[0], n_mid, 1, 1))

    # Generate a batch of images
    gen_images = generator(vector_z)

    # Loss measures generator's ability to fool the discriminator
    g_loss = adversarial_loss(discriminator(gen_images), valid)

    return g_loss, gen_images


def discriminator_forward(real_images, gen_images, valid, fake):
    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(real_images), valid)
    fake_loss = adversarial_loss(discriminator(gen_images), fake)
    d_loss = (real_loss + fake_loss) / 2
    return d_loss


@ms_function
def train_step(images, learning_rate=0.0001, beta_1=0.5):
    optimizer_d = nn.Adam(discriminator.trainable_params(),
                          learning_rate=learning_rate,
                          beta1=beta_1)

    optimizer_g = nn.Adam(generator.trainable_params(),
                          learning_rate=learning_rate,
                          beta1=beta_1)

    optimizer_g.update_parameters_name('optim_g.')
    optimizer_d.update_parameters_name('optim_d')

    grad_generator_fn = ops.value_and_grad(generator_forward,
                                           None,
                                           optimizer_g.parameters,
                                           has_aux=True)
    grad_discriminator_fn = ops.value_and_grad(discriminator_forward,
                                               None,
                                               optimizer_d.parameters)

    valid = ops.ones((images.shape[0], 1), mindspore.float32)
    fake = ops.zeros((images.shape[0], 1), mindspore.float32)

    (g_loss, gen_images), g_grads = grad_generator_fn(images, valid)
    g_loss = ops.depend(g_loss, optimizer_g(g_grads))
    d_loss, d_grads = grad_discriminator_fn(images, gen_images, valid, fake)
    d_loss = ops.depend(d_loss, optimizer_d(d_grads))

    return g_loss, d_loss, gen_images


def show_loss(g_losses, d_losses, storage_path):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G", color='blue')
    plt.plot(d_losses, label="D", color='orange')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(storage_path + "/loss.jpg")


def show_fixed_noise(image_list, storage_path):
    show_list = []
    fig = plt.figure(figsize=(8, 3), dpi=120)
    for epoch, _ in enumerate(image_list):
        images = []
        for i in range(3):
            row = np.concatenate((image_list[epoch][i * 8:(i + 1) * 8]), axis=1)
            images.append(row)
        image = np.clip(np.concatenate((images[:]), axis=0), 0, 1)
        plt.axis("off")
        show_list.append([plt.imshow(image)])

    ani = animation.ArtistAnimation(fig, show_list, interval=1000, repeat_delay=1000, blit=True)
    ani.save(storage_path + './dcgan.gif', writer='pillow', fps=1)


def training():
    # 获取处理后的数据集
    data = create_dataset_imagenet(args_opt.data_url, batch_size=128)
    # 获取数据集大小
    size = data.get_dataset_size()
    print("dataset_size: ", size)
    preview_data(data, storage_path=args_opt.output_path)

    # 创建迭代器
    generator_losses = []
    discriminator_losses = []
    image_list = []

    # 开始循环训练
    print("Starting Training Loop...")

    # 创建存放ckpt的文件夹
    folder = args_opt.output_path + "/ckpt"
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"{folder}创建成功!")

    num_epochs = 10
    for epoch in range(num_epochs):
        # 为每轮训练读入数据
        image = None
        for i, (image, ) in enumerate(data.create_tuple_iterator()):
            g_loss, d_loss, _ = train_step(image, learning_rate=0.0001)
            if i % 50 == 0 or i == size - 1:
                # 输出训练记录
                print('[%2d/%d][%3d/%d]   Loss_D:%7.4f  Loss_G:%7.4f' % (
                    epoch + 1, num_epochs, i + 1, size, d_loss.asnumpy(), g_loss.asnumpy()))
            discriminator_losses.append(d_loss.asnumpy())
            generator_losses.append(g_loss.asnumpy())

        # 每个epoch结束后，使用生成器生成一组图片
        if image:
            fixed_noise = ops.StandardNormal()((image.shape[0], n_mid, 1, 1))
            generate_image = generator(fixed_noise)
            image_list.append(generate_image.transpose(0, 2, 3, 1).asnumpy())

            # 保存网络模型参数为ckpt文件
            mindspore.save_checkpoint(generator,
                                      folder + f"/generator_{epoch}.ckpt")
            mindspore.save_checkpoint(discriminator,
                                      folder + f"/discriminator_{epoch}.ckpt")

    show_loss(generator_losses,
              discriminator_losses,
              storage_path=args_opt.output_path)
    show_fixed_noise(image_list,
                     storage_path=args_opt.output_path)


if __name__ == '__main__':
    args_opt = parse_args()
    np.random.seed(1)
    print(mindspore.run_check())
    os.system("nvidia-smi")

    n_mid = 100  # 隐向量的长度
    adversarial_loss = nn.BCELoss(reduction='mean')
    weight_init = Normal(mean=0, sigma=0.02)
    gamma_init = Normal(mean=1, sigma=0.02)

    generator = Generator()
    generator.set_train()
    # 实例化判别器
    discriminator = Discriminator()
    discriminator.set_train()

    training()
