#! -*- coding: utf-8 -*-

import numpy as np
from scipy import misc
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback
from keras.datasets import cifar10
from keras.datasets.cifar import load_batch
import imageio
from keras import metrics


# load data
x_train, y_train = load_batch("/Users/bai/.keras/datasets/cifar-10-batches-py/data_batch_1")
x_train = np.stack(x_train).transpose(0, 2, 3, 1).astype(np.float32)/255.  # (10000,32,32,3)

height, width, img_dim = 32, 32, 32
center_height = int((height - width) / 2)
z_dim = 128

# Encoder结构
x_in = Input(shape=(img_dim, img_dim, 3), dtype='float32')             # None,32,32,3
x = x_in
x = Conv2D(32, kernel_size=(3,3), strides=(2,2), padding='SAME')(x)    # None,16,16,32
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='SAME')(x)    # None,8,8,64
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = Conv2D(128, kernel_size=(3,3), strides=(2,2), padding='SAME')(x)   # None,4,4,128
x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
x = GlobalAveragePooling2D()(x)                                        # None,128

encoder = Model(x_in, x)
encoder.summary()
map_size = K.int_shape(encoder.layers[-2].output)[1:-1]   # (4,4)

# 解码层，也就是生成器部分
z_in = Input(shape=K.int_shape(x)[1:])                # None,128
z = z_in
z = Dense(4*4*128)(z)                 # None,4*4*128 
z = Reshape((4,4,128,))(z)                   # None,4,4,128
z = Conv2DTranspose(64, kernel_size=(3,3), strides=(2,2), padding='SAME')(z)  # None,8,8,64
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(32, kernel_size=(3,3), strides=(2,2), padding='SAME')(z)  # None,16,16,32
z = BatchNormalization()(z)
z = Activation('relu')(z)
z = Conv2DTranspose(3, kernel_size=(3,3), strides=(2,2), padding='SAME')(z)   # None,32,32,3
z = Activation('sigmoid')(z)

decoder = Model(z_in, z)
decoder.summary()

# 采样
class ScaleShift(Layer):
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)

    def call(self, inputs):
        u, shift, log_scale = inputs
        ux = K.exp(log_scale) * u + shift
        return ux

z_shift = Dense(z_dim)(x)               # None, 128  mu
z_log_scale = Dense(z_dim)(x)           # None, 128  log(sigma)
u = Lambda(lambda t: K.random_normal(shape=K.shape(t)))(z_shift)   # 标准正态分布样本 (None,128)
ux = ScaleShift()([u, z_shift, z_log_scale])   # 生成对应于 z_shift,z_log_scale的样本 (None,128)
x_recon = decoder(ux)

#
recon_loss = 32*32*3*metrics.binary_crossentropy(K.flatten(x_recon), K.flatten(x_in))
kl_loss = K.sum(-0.5 * K.sum(1+2.*z_log_scale - K.square(z_shift) - K.exp(2*z_log_scale), axis=-1))
vae_loss = recon_loss + kl_loss

vae = Model(x_in, x_recon)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(1e-2))

def sample(path):
    n = 9
    figure = np.zeros((img_dim*n, img_dim*n, 3))
    for i in range(n):
        for j in range(n):
            x_recon = decoder.predict(np.random.randn(1, *K.int_shape(x)[1:]))*255.
            digit = x_recon[0]
            figure[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
    figure = (figure + 1) / 2 * 255
    imageio.imwrite(path, figure)

class Evaluate(Callback):
    def __init__(self):
        import os
        self.lowest = 1e10
        self.losses = []
        if not os.path.exists('vae-cifar10'):
            os.mkdir('vae-cifar10')
    def on_epoch_end(self, epoch, logs=None):
        path = 'vae-cifar10/test_%s.png' % epoch
        sample(path)
        self.losses.append((epoch, logs['loss']))
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            encoder.save_weights('vae-cifar10/best_encoder.weights')


evaluator = Evaluate()
vae.fit(x_train,
        shuffle=True,
        nb_epoch=100,
        batch_size=128,
        callbacks=[evaluator])
