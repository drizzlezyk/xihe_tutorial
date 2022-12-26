import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import mindspore
from PIL import Image
from mindspore import nn, ops, context
from mindvision import dataset

nc = 3                    # 图像彩色通道数
nz = 100                  # 隐向量的长度
ngf = 64                  # 特征图在生成器中的大小

class Generator(nn.Cell):
    """DCGAN网络生成器"""

    def __init__(self):
        super(Generator, self).__init__()
        self.generator = nn.SequentialCell(
                nn.Conv2dTranspose(nz, ngf * 8, 4, 1, 'valid'),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf * 8, ngf * 4, 4, 2, 'pad', 1),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf * 4, ngf * 2, 4, 2, 'pad', 1),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf * 2, ngf, 4, 2, 'pad', 1),
                nn.BatchNorm2d(ngf),
                nn.ReLU(),
                nn.Conv2dTranspose(ngf, nc, 4, 2, 'pad', 1),
                nn.Tanh()
            )

    def construct(self, x):
        return self.generator(x)

# 实例化生成器
generator = Generator()
# 从文件中获取模型参数并加载到网络中
param_dict = mindspore.load_checkpoint("./Generator.ckpt", generator)

def generate_image(row_inputs=1, column_inputs=1):
    row_inputs = int(row_inputs)
    column_inputs = int(column_inputs)
    fixed_noise = ops.StandardNormal()((row_inputs * column_inputs, nz, 1, 1))
    img64 = generator(fixed_noise).transpose(0, 2, 3, 1).asnumpy()
    fig = plt.figure(figsize=(row_inputs, column_inputs), dpi=120)
    images = []
    for i in range(column_inputs):
        images.append(np.concatenate((img64[i * row_inputs:(i + 1) * row_inputs]), axis=1))
    img = np.clip(np.concatenate((images[:]), axis=0), 0, 1)
    plt.axis("off")
    plt.imshow(img)
    plt.savefig("predict.jpg")
    return Image.open("predict.jpg")
  
row_inputs = gr.inputs.Number(5, label="the number of columns for images(number>=1)")
column_inputs = gr.inputs.Number(5, label="the number of rows for images(number>=1)")
outputs = gr.Image(label="generated image", shape=(10, 10))

title = "Generate Animer"
description = "Generate a number of images"

demo = gr.Interface(
    fn=generate_image,
    inputs=[row_inputs, column_inputs],
    outputs=outputs,
    title=title,
    description=description,
)

if __name__ == "__main__":
    demo.launch()

    