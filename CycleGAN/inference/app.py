import cv2
from mindspore import load_checkpoint, load_param_into_net
from mindspore import Tensor

import numpy as np
import gradio as gr
from utils.args import get_args
from models.cycle_gan import get_generator
import mindspore.dataset.vision.c_transforms as C

import mindspore
print(mindspore.version)

img_size = 256


def image_preprocessing(img):
    img = cv2.resize(img, (img_size, img_size))

    mean = [0.5 * 255] * 3
    std = [0.5 * 255] * 3
    opn = C.Normalize(mean, std)
    img = opn(img)
    oph = C.HWC2CHW()
    img = oph(img)
    # plt.imshow(img)
    # plt.show()

    # img = img.astype(np.float32)
    # img = img / 255
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.2023, 0.1994, 0.2010])
    # img = (img - mean) / std
    # img = img.astype(np.float32)
    # img = img.transpose(2, 0, 1)

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
        G_B = get_generator(args)
        G_B.set_train(True)
        param_GB = load_checkpoint(
            "./G_B_19.ckpt")
        load_param_into_net(G_B, param_GB)
        fake = G_B(img)
    else:
        G_A = get_generator(args)
        G_A.set_train(True)
        param_GA = load_checkpoint(
            "./G_A_19.ckpt")
        load_param_into_net(G_A, param_GA)
        fake = G_A(img)

    fake = fake.asnumpy()

    # fake = fake[0]
    # fake = fake.transpose(1, 2, 0)
    # fake = (fake - fake.min()) / (fake.max() - fake.min())
    # fake = (fake * 255).astype(np.uint8)

    fake = decode_image(fake)
    # fake = Image.fromarray(fake)
    return fake


examples_vangogh = [["./example_img/1_img_B.jpg", 'Van Gogh->Real'], ["./example_img/vangogh2.jpg", 'Van Gogh->Real']]
examples_real = [["./example_img/1_img_A.jpg", 'Real->Van Gogh'], ["./example_img/nature2.jpg", 'Real->Van Gogh']]
examples = [*examples_real, *examples_vangogh]
gr.Interface(fn=transform,
             title='基于CycleGAN的艺术家画作风格迁移',
             inputs=[gr.inputs.Image(shape=(256, 256), type='numpy'),
                            gr.inputs.Radio(choices=['Real->Van Gogh', 'Van Gogh->Real'],
                                            type='index')],
             outputs=gr.outputs.Image(type='numpy'),
             examples=examples
             ).launch(share=True)
