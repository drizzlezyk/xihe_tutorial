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


import os
import re
import math
import string
import tarfile
import argparse
import six
import numpy as np
import mindspore
import mindspore.dataset as ms_dataset
import mindspore.numpy as mnp

from tqdm import tqdm
from mindspore import load_checkpoint, load_param_into_net, save_checkpoint, \
    set_context, PYNATIVE_MODE, Tensor, nn, ops
from mindspore.common.initializer import Uniform, HeUniform


def parse_args():
    parser = argparse.ArgumentParser(description="train lstm",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pretrain_url',
                        type=str,
                        default='./resnet/resnet_01.ckpt',
                        help='the pretrain model path')

    parser.add_argument('--imdb_path',
                        type=str,
                        default='C:\\datasets\\aclImdb_v1.tar.gz',
                        help='imbd file path')

    parser.add_argument('--glove_path',
                        type=str,
                        default='C:\\datasets\\glove.6B.zip',
                        help='glove file path')

    parser.add_argument('--output_path',
                        default='train/save_model/',
                        type=str,
                        help='the path model save path')

    parser.add_argument('--epochs',
                        default=10,
                        type=int,
                        help='training epochs')

    parser.add_argument('--lr',
                        default=0.0001,
                        type=float,
                        help='learning rate')

    return parser.parse_args()


class IMDBData():
    """IMDB数据集加载器

    加载IMDB数据集并处理为一个Python迭代对象。

    """
    label_map = {
        "pos": 1,
        "neg": 0
    }

    def __init__(self, path, mode="train"):
        self.mode = mode
        self.path = path
        self.docs, self.labels = [], []

        self._load("pos")
        self._load("neg")

    def _load(self, label):
        pattern = re.compile(r"aclImdb/{}/{}/.*\.txt$".format(self.mode, label))
        # 将数据加载至内存
        with tarfile.open(self.path) as file:
            for temp in file:
                if pattern.match(temp.name):
                    # 对文本进行分词、去除标点和特殊字符、小写处理
                    self.docs.append(
                        str(file.extractfile(temp).read().
                            rstrip(six.b("\n\r")).
                            translate(None, six.b(string.punctuation)).
                            lower()).split())
                    self.labels.append(
                        [self.label_map[label]])

    def __getitem__(self, idx):
        return self.docs[idx], self.labels[idx]

    def __len__(self):
        return len(self.docs)


def load_imdb(imdb_path):
    imdb_train = ms_dataset.GeneratorDataset(IMDBData(imdb_path, "train"),
                                             column_names=["text", "label"],
                                             shuffle=True)
    imdb_test = ms_dataset.GeneratorDataset(IMDBData(imdb_path, "test"),
                                            column_names=["text", "label"],
                                            shuffle=False)

    return imdb_train, imdb_test


def load_glove(glove_path):
    glove_100d_path = os.path.join(glove_path, 'glove.6B.100d.txt')
    embeddings = []
    tokens = []
    with open(glove_100d_path, encoding='utf-8') as file:
        for glove in file:
            word, embedding = glove.split(maxsplit=1)
            tokens.append(word)
            embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))
    # 添加 <unk>, <pad> 两个特殊占位符对应的embedding
    embeddings.append(np.random.rand(100))
    embeddings.append(np.zeros((100,), np.float32))

    vocab = ms_dataset.text.Vocab.from_list(tokens,
                                            special_tokens=["<unk>", "<pad>"],
                                            special_first=False)
    embeddings = np.array(embeddings).astype(np.float32)
    return vocab, embeddings


class RNN(nn.Cell):
    def __init__(self, embeddings, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
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
        self.fc = nn.Dense(hidden_dim * 2,
                           output_dim,
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
        output = self.fc(hidden)
        return self.sigmoid(output)


def train_one_epoch(model, train_dataset, epoch=0):
    model.set_train()
    total = train_dataset.get_dataset_size()
    loss_total = 0
    step_total = 0
    with tqdm(total=total) as temp:
        temp.set_description('Epoch %i' % epoch)
        for i in train_dataset.create_tuple_iterator():
            loss = model(*i)
            loss_total += loss.asnumpy()
            step_total += 1
            temp.set_postfix(loss=loss_total/step_total)
            temp.update(1)


def binary_accuracy(predict, true):
    """
    计算每个batch的准确率
    """

    round_predict = np.around(predict)
    correct = (round_predict == true).astype(np.float32)
    acc = correct.sum() / len(correct)
    return acc


def evaluate(model, test_dataset, criterion, epoch=0):
    total = test_dataset.get_dataset_size()
    epoch_loss = 0
    epoch_acc = 0
    step_total = 0
    model.set_train(False)

    with tqdm(total=total) as temp:
        temp.set_description('Epoch %i' % epoch)
        for i in test_dataset.create_tuple_iterator():
            predictions = model(i[0])
            loss = criterion(predictions, i[1])
            epoch_loss += loss.asnumpy()

            acc = binary_accuracy(predictions.asnumpy(), i[1].asnumpy())
            epoch_acc += acc

            step_total += 1
            temp.set_postfix(loss=epoch_loss/step_total, acc=epoch_acc/step_total)
            temp.update(1)

    return epoch_loss / total


def data_preprocessing(vocab, imdb_train, imdb_test):
    lookup_op = ms_dataset.text.Lookup(vocab, unknown_token='<unk>')
    pad_op = ms_dataset.transforms.c_transforms.PadEnd([500],
                                                       pad_value=vocab.tokens_to_ids('<pad>'))
    type_cast_op = ms_dataset.transforms.c_transforms.TypeCast(mindspore.float32)

    imdb_train = imdb_train.map(operations=[lookup_op, pad_op], input_columns=['text'])
    imdb_train = imdb_train.map(operations=[type_cast_op], input_columns=['label'])

    imdb_test = imdb_test.map(operations=[lookup_op, pad_op], input_columns=['text'])
    imdb_test = imdb_test.map(operations=[type_cast_op], input_columns=['label'])

    imdb_train, imdb_valid = imdb_train.split([0.7, 0.3])

    imdb_train = imdb_train.batch(64, drop_remainder=True)
    imdb_valid = imdb_valid.batch(64, drop_remainder=True)

    return imdb_train, imdb_test, imdb_valid


def training_imdb(net, loss, imdb_train, imdb_valid, ckpt_file_name, learning_rate):
    net_with_loss = nn.WithLossCell(net, loss)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=learning_rate)
    model_one_step = nn.TrainOneStepCell(net_with_loss, optimizer)

    best_valid_loss = float('inf')
    for epoch in range(args_opt.epochs):
        train_one_epoch(model_one_step, imdb_train, epoch)

        valid_loss = evaluate(net, imdb_valid, loss, epoch)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(net, ckpt_file_name)


def predict_sentiment(model, vocab, sentence, score_map):
    model.set_train(False)
    tokenized = sentence.lower().split()
    indexed = vocab.tokens_to_ids(tokenized)
    tensor = Tensor(indexed, mindspore.int32)
    tensor = tensor.expand_dims(0)
    prediction = model(tensor)
    return score_map[int(np.round(prediction.asnumpy()))]


def traning_process(imdb_train, imdb_valid, vocab, embeddings):
    # setting some hyper parameters
    hidden_size = 256
    output_size = 1
    num_layers = 2
    bidirectional = True
    dropout = 0.5
    pad_idx = vocab.tokens_to_ids('<pad>')
    ckpt_file_path = os.path.join(args_opt.output_path, 'sentiment-analysis.ckpt')
    loss = nn.BCELoss(reduction='mean')

    # construct model
    net = RNN(embeddings, hidden_size, output_size, num_layers, bidirectional, dropout, pad_idx)

    # use imdb for training and save model
    training_imdb(net, loss, imdb_train, imdb_valid, ckpt_file_path, args_opt.lr)


def evaluate_process(imdb_test, vocab, embeddings):
    # load model
    hidden_size = 256
    output_size = 1
    num_layers = 2
    bidirectional = True
    dropout = 0.5
    pad_idx = vocab.tokens_to_ids('<pad>')
    loss = nn.BCELoss(reduction='mean')

    net = RNN(embeddings, hidden_size, output_size, num_layers, bidirectional, dropout, pad_idx)

    ckpt_file_path = os.path.join(args_opt.output_path, 'sentiment-analysis.ckpt')
    param_dict = load_checkpoint(ckpt_file_path)
    load_param_into_net(net, param_dict)

    evaluate(net, imdb_test, loss)
    score_map = {
        1: "Positive",
        0: "Negative"
    }
    predict_sentiment(net, vocab, "This film is terrible", score_map)


def lstm_for_imdb():
    # load data
    imdb_train, imdb_test = load_imdb(args_opt.imdb_path)
    vocab, embeddings = load_glove(args_opt.glove_path)
    imdb_train, imdb_test, imdb_valid = data_preprocessing(vocab, imdb_train, imdb_test)

    # training
    traning_process(imdb_train, imdb_valid, vocab, embeddings)

    # evaluate
    evaluate_process(imdb_test, vocab, embeddings)


if __name__ == '__main__':
    set_context(mode=PYNATIVE_MODE)
    args_opt = parse_args()
    lstm_for_imdb()
