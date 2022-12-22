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
""" LSTM inference """


import math
import numpy as np

import gradio as gr
import mindspore
import mindspore.numpy as mnp
from mindspore import Tensor, nn, \
    load_checkpoint, load_param_into_net, ops, dataset
from mindspore.common.initializer import Uniform, HeUniform


def load_glove():
    embeddings = []
    tokens = []
    with open("./glove.6B.100d.txt", encoding='utf-8') as file:
        for glove in file:
            word, embedding = glove.split(maxsplit=1)
            tokens.append(word)
            embeddings.append(np.fromstring(embedding,
                                            dtype=np.float32,
                                            sep=' '))
    # 添加 <unk>, <pad> 两个特殊占位符对应的embedding
    embeddings.append(np.random.rand(100))
    embeddings.append(np.zeros((100,), np.float32))

    vocab = dataset.text.Vocab.from_list(tokens,
                                         special_tokens=["<unk>", "<pad>"],
                                         special_first=False)
    embeddings = np.array(embeddings).astype(np.float32)
    return vocab, embeddings


class RNN(nn.Cell):
    def __init__(self, embeddings, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding(vocab_size, embedding_dim,
                                      embedding_table=Tensor(embeddings),
                                      padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
        self.fc_layer = nn.Dense(hidden_dim * 2, output_dim,
                           weight_init=weight_init,
                           bias_init=bias_init)
        self.dropout = nn.Dropout(1 - dropout)
        self.sigmoid = ops.Sigmoid()

    def construct(self, inputs):
        embedded = self.dropout(self.embedding(inputs))
        _, (hidden, _) = self.rnn(embedded)
        hidden = self.dropout(mnp.concatenate((hidden[-2, :, :],
                                               hidden[-1, :, :]),
                                              axis=1))
        output = self.fc_layer(hidden)
        return self.sigmoid(output)


def predict_sentiment(model, vocab, sentence):
    model.set_train(False)
    tokenized = sentence.lower().split()
    indexed = vocab.tokens_to_ids(tokenized)
    tensor = mindspore.Tensor(indexed, mindspore.int32)
    tensor = tensor.expand_dims(0)
    prediction = model(tensor)
    return prediction.asnumpy()


vocab, embeddings = load_glove()

net = RNN(embeddings,
          hidden_dim=256,
          output_dim=1,
          n_layers=2,
          bidirectional=True,
          dropout=0.5,
          pad_idx=vocab.tokens_to_ids('<pad>'))
# 将模型参数存入parameter的字典中
param_dict = load_checkpoint("./sentiment-analysis.ckpt")

# 将参数加载到网络中
load_param_into_net(net, param_dict)


def predict_emotion(sentence):
    # 预测
    pred = predict_sentiment(net, vocab, sentence).item()
    result = {
        "Positive 🙂": pred,
        "Negative 🙃": 1 - pred,
    }
    return result


gr.Interface(
    fn=predict_emotion,
    inputs=gr.inputs.Textbox(
        lines=3,
        placeholder="Type a phrase that has some emotion",
        label="Input Text",
    ),
    outputs="label",
    title="基于LSTM的文本情感分类任务",
    examples=[
        "This film is terrible",
        "This film is great",
    ],
).launch(share=True)
