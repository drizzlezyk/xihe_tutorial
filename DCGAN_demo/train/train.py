import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import mindspore
import mindspore.dataset as ds
import mindspore.dataset.vision as vision

from mindspore import context
from mindspore import nn, ops
from mindspore import ms_function
from mindspore.common.initializer import Normal
from mindvision import dataset


# 选择执行模式为图模式；指定训练使用的平台为"GPU"，如需使用昇腾硬件可将其替换为"Ascend"
# context.set_context(mode=context.PYNATIVE_MODE)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
# ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
# data_root = "./datasets"  # 数据集根目录
batch_size = 128          # 批量大小
image_size = 64           # 训练图像空间大小
nc = 3                    # 图像彩色通道数
nz = 100                  # 隐向量的长度
ngf = 64                  # 特征图在生成器中的大小
ndf = 64                  # 特征图在判别器中的大小
num_epochs = 10           # 训练周期数
lr = 0.0002               # 学习率
beta1 = 0.5               # Adam优化器的beta1超参数

def parse_args():
    # 创建解析
    parser = argparse.ArgumentParser(description="train dcgan",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 添加参数
    parser.add_argument('--pretrain_url', type=str, default='', help='the pretrain model path')
    parser.add_argument('--data_url', type=str, default='', help='the training data path')
    parser.add_argument('--output_path', default='', type=str, help='the path model saved')
    # 解析参数
    args_opt = parser.parse_args()
    return args_opt

def create_dataset_imagenet(dataset_path):
    """数据加载"""
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
    # plt.show()
    plt.savefig(storage_path + "/preview.jpg")

weight_init = Normal(mean=0, sigma=0.02)
gamma_init = Normal(mean=1, sigma=0.02)

class Generator(nn.Cell):
    """DCGAN网络生成器"""

    def __init__(self):
        super(Generator, self).__init__()
        self.generator = nn.SequentialCell(
                nn.Conv2dTranspose(nz, ngf * 8, 4, 1, 'valid', weight_init=weight_init),
                nn.BatchNorm2d(ngf * 8, gamma_init=gamma_init),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf * 8, ngf * 4, 4, 2, 'pad', 1, weight_init=weight_init),
                nn.BatchNorm2d(ngf * 4, gamma_init=gamma_init),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf * 4, ngf * 2, 4, 2, 'pad', 1, weight_init=weight_init),
                nn.BatchNorm2d(ngf * 2, gamma_init=gamma_init),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf * 2, ngf, 4, 2, 'pad', 1, weight_init=weight_init),
                nn.BatchNorm2d(ngf, gamma_init=gamma_init),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf, nc, 4, 2, 'pad', 1, weight_init=weight_init),
                nn.Tanh()
            )

    def construct(self, x):
        return self.generator(x)

# 实例化生成器
generator = Generator()
generator.set_train()

class Discriminator(nn.Cell):
    """DCGAN网络判别器"""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.SequentialCell(
                nn.Conv2d(nc, ndf, 4, 2, 'pad', 1, weight_init=weight_init),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 'pad', 1, weight_init=weight_init),
                nn.BatchNorm2d(ngf * 2, gamma_init=gamma_init),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 'pad', 1, weight_init=weight_init),
                nn.BatchNorm2d(ngf * 4, gamma_init=gamma_init),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 'pad', 1, weight_init=weight_init),
                nn.BatchNorm2d(ngf * 8, gamma_init=gamma_init),
                nn.LeakyReLU(0.2),
                nn.Conv2d(ndf * 8, 1, 4, 1, 'valid', weight_init=weight_init),
            )
        self.adv_layer = nn.Sigmoid()

    def construct(self, x):
        out = self.discriminator(x)
        out = out.reshape(out.shape[0], -1)
        return self.adv_layer(out)

# 实例化判别器
discriminator = Discriminator()
discriminator.set_train()

# 定义损失函数
adversarial_loss = nn.BCELoss(reduction='mean')

# 创建一批隐向量用来观察G
np.random.seed(1)

optimizer_D = nn.Adam(discriminator.trainable_params(), learning_rate=lr, beta1=beta1)
optimizer_G = nn.Adam(generator.trainable_params(), learning_rate=lr, beta1=beta1)
optimizer_G.update_parameters_name('optim_g.')
optimizer_D.update_parameters_name('optim_d')

def generator_forward(real_imgs, valid):
    # Sample noise as generator input
    z = ops.StandardNormal()((real_imgs.shape[0], nz, 1, 1))

    # Generate a batch of images
    gen_imgs = generator(z)

    # Loss measures generator's ability to fool the discriminator
    g_loss = adversarial_loss(discriminator(gen_imgs), valid)

    return g_loss, gen_imgs

def discriminator_forward(real_imgs, gen_imgs, valid, fake):
    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(gen_imgs), fake)
    d_loss = (real_loss + fake_loss) / 2
    return d_loss

grad_generator_fn = ops.value_and_grad(generator_forward, None,
                                        optimizer_G.parameters,
                                        has_aux=True)
grad_discriminator_fn = ops.value_and_grad(discriminator_forward, None,
                                           optimizer_D.parameters)    

@ms_function
def train_step(imgs):
    valid = ops.ones((imgs.shape[0], 1), mindspore.float32)
    fake = ops.zeros((imgs.shape[0], 1), mindspore.float32)

    (g_loss, gen_imgs), g_grads = grad_generator_fn(imgs, valid)
    g_loss = ops.depend(g_loss, optimizer_G(g_grads))
    d_loss, d_grads = grad_discriminator_fn(imgs, gen_imgs, valid, fake)
    d_loss = ops.depend(d_loss, optimizer_D(d_grads))

    return g_loss, d_loss, gen_imgs



def show_loss(G_losses, D_losses, storage_path):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G", color='blue')
    plt.plot(D_losses, label="D", color='orange')
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    # plt.show()
    plt.savefig(storage_path + "/loss.jpg")

def show_fixed_noise(image_list, storage_path):
    import matplotlib.animation as animation
    show_list = []
    fig = plt.figure(figsize=(8, 3), dpi=120)
    for epoch in range(len(image_list)):
        images = []
        for i in range(3):
            row = np.concatenate((image_list[epoch][i * 8:(i + 1) * 8]), axis=1)
            images.append(row)
        img = np.clip(np.concatenate((images[:]), axis=0), 0, 1)
        plt.axis("off")
        show_list.append([plt.imshow(img)])

    ani = animation.ArtistAnimation(fig, show_list, interval=1000, repeat_delay=1000, blit=True)
    ani.save(storage_path + './dcgan.gif', writer='pillow', fps=1)

def main(data_root, storage_path):
    # 获取处理后的数据集
    data = create_dataset_imagenet(data_root)
    # 获取数据集大小
    size = data.get_dataset_size()
    print("dataset_size: ", size)
    preview_data(data, storage_path=storage_path)
              
    # 创建迭代器
    data_loader = data.create_dict_iterator(output_numpy=True, num_epochs=num_epochs)
    G_losses = []
    D_losses = []
    image_list = []

    # 开始循环训练
    print("Starting Training Loop...")

    # 创建存放ckpt的文件夹
    folder = storage_path + f"/ckpt"
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"{folder}创建成功!")
    for epoch in range(num_epochs):
        # 为每轮训练读入数据
        for i, (imgs, ) in enumerate(data.create_tuple_iterator()):
            g_loss, d_loss, gen_imgs = train_step(imgs)
            if i % 50 == 0 or i == size - 1:
                # 输出训练记录
                print('[%2d/%d][%3d/%d]   Loss_D:%7.4f  Loss_G:%7.4f' % (
                    epoch + 1, num_epochs, i + 1, size, d_loss.asnumpy(), g_loss.asnumpy()))
            D_losses.append(d_loss.asnumpy())
            G_losses.append(g_loss.asnumpy())

        # 每个epoch结束后，使用生成器生成一组图片
        fixed_noise = ops.StandardNormal()((imgs.shape[0], nz, 1, 1))
        img = generator(fixed_noise)
        image_list.append(img.transpose(0, 2, 3, 1).asnumpy())

        # 保存网络模型参数为ckpt文件
        mindspore.save_checkpoint(generator, folder + f"/generator_{epoch}.ckpt")
        mindspore.save_checkpoint(discriminator, folder + f"/discriminator_{epoch}.ckpt")
    # # 保存网络模型参数为ckpt文件
    # mindspore.save_checkpoint(generator, storage_path + "ckpt/Generator.ckpt")
    # mindspore.save_checkpoint(discriminator, storage_path + "ckpt/Discriminator.ckpt")

    show_loss(G_losses, D_losses, storage_path=storage_path)
    show_fixed_noise(image_list, storage_path=storage_path)

if __name__ == '__main__':
    print(mindspore.run_check())
    os.system("nvidia-smi")
    args_opt = parse_args()
    data_root = args_opt.data_url
    storage_path = args_opt.output_path
    ckpt_path = args_opt.pretrain_url
    main(data_root, storage_path)