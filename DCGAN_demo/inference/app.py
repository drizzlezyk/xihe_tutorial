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

""" DCGAN inference """


import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import mindspore
from PIL import Image
from mindspore import nn, ops


class Generator(nn.Cell):
    """DCGAN网络生成器"""

    def __init__(self, n_chanel=3, n_mid=100, ngf_layer=64):
        super().__init__()
        self.generator = nn.SequentialCell(
                nn.Conv2dTranspose(n_mid, ngf_layer * 8, 4, 1, 'valid'),
                nn.BatchNorm2d(ngf_layer * 8),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf_layer * 8, ngf_layer * 4, 4, 2, 'pad', 1),
                nn.BatchNorm2d(ngf_layer * 4),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf_layer * 4, ngf_layer * 2, 4, 2, 'pad', 1),
                nn.BatchNorm2d(ngf_layer * 2),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf_layer * 2, ngf_layer, 4, 2, 'pad', 1),
                nn.BatchNorm2d(ngf_layer),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf_layer, n_chanel, 4, 2, 'pad', 1),
                nn.Tanh()
            )

    def construct(self, input_vector):
        return self.generator(input_vector)


def generate_image(row_inputs=1, column_inputs=1):
    row_inputs = int(row_inputs)
    column_inputs = int(column_inputs)
    fixed_noise = ops.StandardNormal()((row_inputs * column_inputs, 100, 1, 1))
    img64 = generator(fixed_noise).transpose(0, 2, 3, 1).asnumpy()
    plt.figure(figsize=(row_inputs, column_inputs), dpi=120)
    images = []
    for i in range(column_inputs):
        images.append(np.concatenate((img64[i * row_inputs:(i + 1) * row_inputs]), axis=1))
    img = np.clip(np.concatenate((images[:]), axis=0), 0, 1)
    plt.axis("off")
    plt.imshow(img)
    plt.savefig("predict.jpg")
    return Image.open("predict.jpg")


# load model
generator = Generator()
param_dict = mindspore.load_checkpoint("./Generator.ckpt", generator)


TITLE = "Generate Animer"
DESC = "Generate a number of images"

demo = gr.Interface(
    fn=generate_image,
    inputs=[gr.inputs.Number(5, label="the number of columns for images(number>=1)"),
            gr.inputs.Number(5, label="the number of rows for images(number>=1)")],
    outputs=gr.Image(label="generated image", shape=(10, 10)),
    title=TITLE,
    description=DESC,
)


if __name__ == "__main__":
    demo.launch()
