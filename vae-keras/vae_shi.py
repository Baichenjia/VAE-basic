#! -*- coding:utf-8 -*-
# 一个简单的基于VAE和CNN的作诗机器人
# 来自：https://kexue.fm/archives/5332

import os
import re
import codecs
import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import Callback

n = 5            # 只抽取五言诗
latent_dim = 64  # 隐变量维度
hidden_dim = 64  # 词向量的维度

s = codecs.open('shi.txt', encoding='utf-8').read()

# 通过正则表达式找出所有的五言诗
s = re.findall(u'　　(.{%s}，.{%s}。.*?)\r\n'%(n,n), s)
shi = []
for i in s:
    for j in i.split(u'。'): # 按句切分
        if j:
            shi.append(j)

shi = np.array([i[:n] + i[n+1:] for i in shi if len(i) == 2*n+1])
# print(shi[0])  # 秦川雄帝宅，函谷壮皇居

# 建立每个字的id
id2char = dict(enumerate(set(''.join(shi))))
char2id = {j:i for i,j in id2char.items()}
# print(char2id["秦"])   # 5579
# print(id2char[5579])   # 秦

# 诗歌id化
shi2id = [[char2id[j] for j in i] for i in shi]
shi2id = np.array(shi2id)   # shape=(148410, 10)，每行是一句话

# Gated CNN，一个卷积输出为sigmoid，另外一个正常，二者相乘
# 将一个 (None,seq_len,embedding_len) 映射成 (None, seq_len, output_dim*2)
# 随后通过 Gated 将最后一维分成两本相乘，输出为 (None, seq_len, output_dim)
# seq_len 不变是因为设置中 padding='same'
class GCNN(Layer):
    def __init__(self, output_dim=None, residual=False, **kwargs):
        super(GCNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.residual = residual

    def build(self, input_shape):
        if self.output_dim == None:   # 默认输入输出维度相同 
            self.output_dim = input_shape[-1]
        # 输出为output_dim*2，其中一半经过sigmoid后作为gate
        # filter_size=3, input channels=input_shape[-1]
        self.kernel = self.add_weight(name='gcnn_kernel',
                                      shape=(3, input_shape[-1], self.output_dim*2),
                                      initializer='glorot_uniform',
                                      trainable=True)

    def call(self, x):
        # x.shape=(batch, in_width, in_channels)
        _ = K.conv1d(x, self.kernel, padding='same')
        # 输出的一半作为gate，乘以另外一半
        _ = _[:, :, :self.output_dim] * K.sigmoid(_[:, :, self.output_dim:])
        if self.residual:
            return _ + x
        else:
            return _

# 输入的每句话包含10个字  (None,10)
input_sentence = Input(shape=(2 * n, ), dtype='int32')

# Embedding层， 输入为 词表长度、词向量的维度. 这里词表长度为 len(char2id)
# 词向量维度为 hidden_dim=64. 将input_setence=(None,10)作为输入，输出为(None,10,64)
input_vec = Embedding(len(char2id), hidden_dim)(input_sentence)

# 经过两层GCNN层，但参数output_dim设置None，所以输出维度与输入维度相等，是(None,10,64)
h = GCNN(residual=True)(input_vec)
h = GCNN(residual=True)(h)

# 输入为(None,10,64),输出为(None,64)，相当于对所有词进行了average
h = GlobalAveragePooling1D()(h)

# h为输入特征，将其映射为均值和方差.
z_mean = Dense(latent_dim)(h)      # (None,latent_dim)=(None,64)
z_log_var = Dense(latent_dim)(h)   # (None,64)


# Reparameterition Trick. 从 N(0,I) 中采样，映射到 N(z_mean,z_log_var) 中
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
    return z_mean + K.exp(z_log_var / 2) * epsilon


z = Lambda(sampling)([z_mean, z_log_var])   # None,64

# 定义解码层，分开定义是为了后面的重用
decoder_hidden = Dense(hidden_dim*(2*n))
decoder_cnn = GCNN(residual=True)
decoder_dense = Dense(len(char2id), activation='softmax')

h = decoder_hidden(z)                 # (None,64) -> (None,64*10)
h = Reshape((2*n, hidden_dim))(h)     # (None,64*10) -> (None,10,64) [seq_len,embedding_dim]
h = decoder_cnn(h)                    # (None,10,64) -> (None,10,64)
output = decoder_dense(h)             # (None,10,64) -> (None,10,len(char2id))

# 建立模型
vae = Model(input_sentence, output)

# xent_loss是重构loss，kl_loss是KL loss
xent_loss = K.sum(K.sparse_categorical_crossentropy(input_sentence, output), 1)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

# 重用解码层，构建单独的生成模型
decoder_input = Input(shape=(latent_dim,))
_ = decoder_hidden(decoder_input)
_ = Reshape((2*n, hidden_dim))(_)
_ = decoder_cnn(_)
_output = decoder_dense(_)
generator = Model(decoder_input, _output)


# 利用生成模型随机生成一首诗
def gen():
    r = generator.predict(np.random.randn(5, latent_dim))   # (None,10,6872)
    r = r.argmax(axis=-1)  # (None,10)
    output = []
    for idx in range(r.shape[0]):
        output.append(''.join([id2char[i] for i in r[idx, :n]])+ u'，'+ ''.join(
            [id2char[i] for i in r[idx, n:]]))
    return output

# 回调器，方便在训练过程中输出
class Evaluate(Callback):
    def __init__(self):
        self.log = []

    def on_epoch_end(self, epoch, logs=None):
        # 在每个训练周期结束时，调用 gen() 函数，观察诗生成的效果
        output = gen()
        self.log.append(output[-1])
        for out in output:
            print(out)


evaluator = Evaluate()
if not os.path.exists("vae-shi/vae-shi.h5"):
    vae.fit(shi2id,
            shuffle=True,
            epochs=50,
            batch_size=512,
            callbacks=[evaluator])
    vae.save_weights('vae-shi/vae-shi.h5')
    print("save done.")
else:
    vae.load_weights("vae-shi/vae-shi.h5")
    print("--\n load weights from local.. done.")

for i in range(20):
    print("--- ", i)
    print(gen()[-1])
