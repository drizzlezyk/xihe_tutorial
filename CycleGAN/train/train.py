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

"""General-purpose training script for image-to-image translation.
You need to specify the dataset ('--data_path'),
and model ('--G_A_ckpt', '--G_B_ckpt', '--D_A_ckpt', '--D_B_ckpt').
Example:
    Train a resnet model:
        python train.py --data_path ./data/horse2zebra --G_A_ckpt ./G_A.ckpt
"""

import mindspore as ms

from mindspore import nn
from mindspore.profiler.profiling import Profiler

from utils.args import get_args
from utils.reporter import Reporter
from utils.tools import get_lr, ImagePool, load_ckpt
from dataset.cyclegan_dataset import create_dataset
from models.losses import DiscriminatorLoss, GeneratorLoss
from models.cycle_gan import get_generator, get_discriminator, \
    Generator, TrainOneStepG, TrainOneStepD

ms.set_seed(1)


def load_model(args):
    """
    load Generator A, B and Discriminator A, B
    :param args:
    :return: net_generator, net_discriminator
    """
    generator_a = get_generator(args)
    generator_b = get_generator(args)
    discriminator_a = get_discriminator(args)
    discriminator_b = get_discriminator(args)
    if args.load_ckpt:
        load_ckpt(args, generator_a, generator_b, discriminator_a, discriminator_b)

    generator = Generator(generator_a, generator_b, args.lambda_idt > 0)

    loss_discriminator = DiscriminatorLoss(args,
                                           discriminator_a,
                                           discriminator_b)
    loss_generator = GeneratorLoss(args,
                                   generator,
                                   discriminator_a,
                                   discriminator_b)

    optimizer_generator = nn.Adam(generator.trainable_params(),
                                  get_lr(args),
                                  beta1=args.beta1)
    optimizer_discriminator = nn.Adam(loss_discriminator.trainable_params(),
                                      get_lr(args),
                                      beta1=args.beta1)

    net_generator = TrainOneStepG(loss_generator, generator, optimizer_generator)
    net_discriminator = TrainOneStepD(loss_discriminator, optimizer_discriminator)
    return net_generator, net_discriminator


def train():
    """Train function."""
    args = get_args("train")
    if args.need_profiler:
        profiler = Profiler(output_path=args.output_path, is_detail=True, is_show_op_path=True)

    dataset = create_dataset(args)

    net_generator, net_discriminator = load_model(args)

    image_pool_a = ImagePool(args.pool_size)
    image_pool_b = ImagePool(args.pool_size)

    if args.rank == 0:
        reporter = Reporter(args)
        reporter.info('==========start training===============')
    for _ in range(args.max_epoch):
        if args.rank == 0:
            reporter.epoch_start()
        for data in dataset.create_dict_iterator():
            res_generator = net_generator(data["image_A"], data["image_B"])
            fake_a, fake_b = res_generator[0], res_generator[1]
            res_discriminator = net_discriminator(data["image_A"],
                                                  data["image_B"],
                                                  image_pool_a.query(fake_a),
                                                  image_pool_b.query(fake_b))
            if args.rank == 0:
                reporter.step_end(res_generator, res_discriminator)
                reporter.visualizer(data["image_A"], data["image_B"], fake_a, fake_b)
        if args.rank == 0:
            reporter.epoch_end(net_generator)
        if args.need_profiler:
            profiler.analyse()
            break
    if args.rank == 0:
        reporter.info('==========end training===============')


if __name__ == "__main__":
    train()
