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

"""CycleGAN inference based on Gradio"""


import cv2
import numpy as np
import gradio as gr

import mindspore.dataset.vision.c_transforms as C
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Tensor

from utils.args import get_args
from models.cycle_gan import get_generator


def image_preprocessing(img):
    img_size = 256
    img = cv2.resize(img, (img_size, img_size))
    mean = [0.5 * 255] * 3
    std = [0.5 * 255] * 3
    opn = C.Normalize(mean, std)
    img = opn(img)
    oph = C.HWC2CHW()
    img = oph(img)
    img = img.reshape(1, 3, img_size, img_size)
    img = Tensor(img)
    return img


def decode_image(img):
    """Decode a [1, C, H, W] Tensor to image numpy array."""
    mean = 0.5 * 255
    std = 0.5 * 255
    return (img[0] * std + mean).astype(np.uint8).transpose((1, 2, 0))


def transform(img, direction):
    img = image_preprocessing(img)
    args = get_args("")
    if direction == 1:
        generator_b = get_generator(args)
        generator_b.set_train(True)
        param_b = load_checkpoint(
            "./G_B_19.ckpt")
        load_param_into_net(generator_b, param_b)
        fake = generator_b(img)
    else:
        generator_a = get_generator(args)
        generator_a.set_train(True)
        param_a = load_checkpoint(
            "./G_A_19.ckpt")
        load_param_into_net(generator_a, param_a)
        fake = generator_a(img)

    fake = fake.asnumpy()
    fake = decode_image(fake)
    return fake


examples_vangogh = [["./example_img/1_img_B.jpg", 'Van Gogh->Real'],
                    ["./example_img/vangogh2.jpg", 'Van Gogh->Real']]
examples_real = [["./example_img/1_img_A.jpg", 'Real->Van Gogh'],
                 ["./example_img/nature2.jpg", 'Real->Van Gogh']]
examples = [*examples_real, *examples_vangogh]

gr.Interface(fn=transform,
             title='基于CycleGAN的艺术家画作风格迁移',
             inputs=[gr.inputs.Image(shape=(256, 256), type='numpy'),
                     gr.inputs.Radio(choices=['Real->Van Gogh',
                                              'Van Gogh->Real'],
                                     type='index')],
             outputs=gr.outputs.Image(type='numpy'),
             examples=examples
             ).launch(share=True)
