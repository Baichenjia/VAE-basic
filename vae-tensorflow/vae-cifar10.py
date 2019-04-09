#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
from keras.datasets.cifar import load_batch
from keras import metrics

tfe = tf.contrib.eager
tf.enable_eager_execution()

# load data
(train_images, _), (test_images, _) = tf.keras.datasets.cifar10.load_data()

test_images = train_images[:10000, :, :, :]

train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.
assert train_images.shape == (50000, 32, 32, 3)
assert test_images.shape == (10000, 32, 32, 3)

# Binarization
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

TRAIN_BUF = 50000
BATCH_SIZE = 100
TEST_BUF = 10000

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation=tf.nn.relu),  # (None,15,15,32)
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation=tf.nn.relu),  # (None,7,7,64)
            tf.keras.layers.Flatten(),                               # (None, 3136)
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),          # (None, 2*latent_dim) 输出 mu 和 sigma
            ]
        )
        self.inference_net.summary()

        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),    
            tf.keras.layers.Dense(units=8*8*32, activation=tf.nn.relu),  # (None,8*8*32)
            tf.keras.layers.Reshape(target_shape=(8, 8, 32)),    # (None,8,8,32)
            tf.keras.layers.Conv2DTranspose(         # (None,16,16,64)
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(         # (None,32,32,32)
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation=tf.nn.relu),
            # No activation
            tf.keras.layers.Conv2DTranspose(         # (None,32,32,3)
                filters=3, kernel_size=3, strides=(1, 1), padding="SAME"),
        ])
        self.generative_net.summary()

    def sample(self, eps=None):
        # 从标准正态分布中采样100个数，随后从decoder映射为图像
        if eps is None:
            eps = tf.random_normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        # 将原始图像x经过 inference_net 映射为 mean 和 logvar
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        # 从标准正态分布中采样 (batch_size,latent_dim) 个数，随后映射
        eps = tf.random_normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        # 将z映射为图像
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


## Define the loss function and the optimizer
# In practice, we optimize the single sample Monte Carlo estimate of this expectation: 
# $$\log p(x| z) + \log p(z) - \log q(z|x),$$
# where $z$ is sampled from $q(z|x)$. 
# **Note**: we could also analytically compute the KL term, but here we incorporate all three terms in the Monte Carlo estimator for simplicity.


def log_normal_pdf(sample, mean, logvar, raxis=1):
    # 可以验证，是高斯公式
    log2pi = tf.log(2. * np.pi)
    return tf.reduce_sum(
        -0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(model, x):
    # 完整的数据流
    mean, logvar = model.encode(x)          # mean.shape=logvar.shape=(batch_size,50)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    # 损失
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


def compute_gradients(model, x):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    return tape.gradient(loss, model.trainable_variables), loss


optimizer = tf.train.AdamOptimizer(1e-4)


def apply_gradients(optimizer, gradients, variables, global_step=None):
    optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


# Training
epochs = 100
latent_dim = 50
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random_normal(
    shape=[num_examples_to_generate, latent_dim])

model = CVAE(latent_dim)


def generate_and_save_images(model, epoch, test_input):
    predictions = model.sample(test_input)
    plt.switch_backend('agg')
    plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('vae-cifar10/image_at_epoch_{:04d}.png'.format(epoch))


generate_and_save_images(model, 0, random_vector_for_generation)

model.load_weights('vae-cifar10/model_weights.h5')
print("load done.")

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        gradients, loss = compute_gradients(model, train_x)
        apply_gradients(optimizer, gradients, model.trainable_variables)

    end_time = time.time()

    model.save_weights('vae-cifar10/model_weights.h5')

    if epoch % 1 == 0:
        loss = tfe.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, ''time elapse for current epoch {}'.format(
                    epoch, elbo, end_time - start_time))
        generate_and_save_images(
            model, epoch, random_vector_for_generation)

