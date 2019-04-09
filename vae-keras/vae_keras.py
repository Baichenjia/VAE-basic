#! -*- coding: utf-8 -*-
# 用Keras实现的VAE
# https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import load_model

batch_size = 100
original_dim = 784
latent_dim = 2             # 隐变量取2维只是为了方便后面画图
intermediate_dim = 256
epochs = 300
num_classes = 10

# 加载MNIST数据集
# x_train.shape = (60000, 28, 28)   y_train_.shape=(60000,)  test为10000个样本
(x_train, y_train_), (x_test, y_test_) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))  # (60000,784)
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))  # (10000,784)
y_train = to_categorical(y_train_, num_classes)  # (60000,10) 转为one-hot
y_test = to_categorical(y_test_, num_classes)    # (10000,10) 转为one-hot

# (None,784) -> (None,256)
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)

# p(Z|X)的均值和方差
z_mean = Dense(latent_dim)(h)      # (None,256) -> (None,2)
z_log_var = Dense(latent_dim)(h)   # (None,256) -> (None,2)


# 重参数技巧
def sampling(args):
    # z_mean.shape=(None,2), z_log_var.shape=(None,2)
    z_mean, z_log_var = args
    # 生成随机数. 均值为0，方差为1.  epsilon.shape=(None,2)
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.0)
    generate_latent = z_mean + K.exp(z_log_var / 2) * epsilon   # mean+std*epsilon
    # generate_latent.shape=(10,2)
    return generate_latent              # shape


# 重参数层，相当于给输入加入噪声
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# 解码层，也就是生成器部分
decoder_h = Dense(intermediate_dim, activation='relu')
h_decoded = decoder_h(z)                                   # (None,2) -> (None,256)

decoder_mean = Dense(original_dim, activation='sigmoid')   # (None,256) -> (None,784)
x_decoded_mean = decoder_mean(h_decoded)

# 建立模型
vae = Model(x, x_decoded_mean)

# xent_loss是重构loss，kl_loss是KL loss. 计算后都是(None,1)的量，每行代表每个样本的损失
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)  # 由于已经规约到0-1，因此使用log-loss
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

# add_loss是新增的方法，用于更灵活地添加各种loss
vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

if not os.path.exists("vae-train/vae-model.h5"):
    vae.fit(x_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))
    vae.save_weights('vae-train/vae-model.h5')
else:
    vae.load_weights("vae-train/vae-model.h5")
    print("--\n load weights from local.. done.")

# 构建encoder，然后观察各个数字在隐空间的分布. 输出的是均值，观察的是均值的分布
encoder = Model(x, z_mean)
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)  # (10000,2)

plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test_)
plt.colorbar()
plt.savefig("vae-train/vae-train-mean.jpg")

# 构建生成器
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

# 观察隐变量的两个维度变化是如何影响输出结果的
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))

#用正态分布的分位数来构建隐变量对
# [-1.64485363, -1.03643339, -0.67448975, -0.38532047, -0.12566135, 0.12566135,  0.38532047,  0.67448975,  1.03643339,  1.64485363])
# 解释: 标准正态分布的0.05分位数再 -1.6. 0.5分位数在0附近. 
# grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

grid_x = np.linspace(-4., 4., n)
grid_y = np.linspace(-4., 4., n)

for i, xi in enumerate(grid_x):
    for j, yi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])           # 采样一个隐变量
        x_decoded = generator.predict(z_sample)   # 解码器解码
        digit = x_decoded[0].reshape(digit_size, digit_size)   # reshape成图片
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit


plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.savefig("vae-train/vae-normal-decoder.jpg")
