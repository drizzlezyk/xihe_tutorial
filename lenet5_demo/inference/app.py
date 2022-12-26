import cv2
import numpy as np
import gradio as gr

from mindspore import load_checkpoint, load_param_into_net
from mindspore import Tensor
from mindvision.classification.models import lenet
from mindspore.train import Model
from mindspore.nn import Softmax

NUM_CLASS = 10


def predict_digit(img):
    # 加载网路
    # 将模型参数存入parameter的字典中
    param_dict = load_checkpoint("./best.ckpt")

    # 重新定义一个LeNet神经网络,注意输入是32*32，loss采用的是SoftmaxCE
    network = lenet(num_classes=NUM_CLASS, pretrained=False)
    # 将参数加载到网络中
    load_param_into_net(network, param_dict)
    model = Model(network)
    
    # 处理图片,转化为 N，C，H,W
    img = cv2.resize(img, (32, 32))
    img = img.astype(np.float32)
    img = img / 255
    img = img.reshape((1, 1, 32, 32))
    
    #预测
    predict_score = model.predict(Tensor(img)).reshape(-1)
    predict_probability = Softmax()(predict_score)
    
    return {str(i): predict_probability[i].asnumpy().item() for i in range(NUM_CLASS)}

gr.Interface(fn=predict_digit, 
             inputs="sketchpad", 
             outputs=gr.outputs.Label(num_top_classes=NUM_CLASS, label="预测类别"),
             live=False,
             css=".footer {display:none !important}",
             title="0-9数字画板",
             description="画0-9数字",
             thumbnail="https://raw.githubusercontent.com/gradio-app/real-time-mnist/master/thumbnail2.png").launch()