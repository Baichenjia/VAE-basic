#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2018 The TensorFlow Authors.
# 
# Licensed under the Apache License, Version 2.0 (the "License").
# 
# # Convolutional VAE: An example with tf.keras and eager
# 
# <table class="tfo-notebook-buttons" align="left"><td>
# <a target="_blank"  href="https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/generative_examples/cvae.ipynb">
#     <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />Run in Google Colab</a>  
# </td><td>
# <a target="_blank"  href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/eager/python/examples/generative_examples/cvae.ipynb"><img width=32px src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />View source on GitHub</a></td></table>

# ![evolution of output during training](https://tensorflow.org/images/autoencoders/cvae.gif)
# 
# This notebook demonstrates how to generate images of handwritten digits using [tf.keras](https://www.tensorflow.org/programmers_guide/keras) and [eager execution](https://www.tensorflow.org/programmers_guide/eager) by training a Variational Autoencoder. (VAE, [[1]](https://arxiv.org/abs/1312.6114), [[2]](https://arxiv.org/abs/1401.4082)).
# 
# 

import tensorflow as tf
tfe = tf.contrib.eager
tf.enable_eager_execution()

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

# ## Load the MNIST dataset
# Each MNIST image is originally a vector of 784 integers, each of which is between 0-255 and represents the intensity of a pixel. 
# We model each pixel with a Bernoulli distribution in our model, and we statically binarize the dataset.

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()
print(train_images.shape, test_images.shape)

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')

# Normalizing the images to the range of [0., 1.]
train_images /= 255.
test_images /= 255.

# Binarization
train_images[train_images >= .5] = 1.
train_images[train_images < .5] = 0.
test_images[test_images >= .5] = 1.
test_images[test_images < .5] = 0.

TRAIN_BUF = 60000
BATCH_SIZE = 100
TEST_BUF = 10000

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(test_images).shuffle(TEST_BUF).batch(BATCH_SIZE)


class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation=tf.nn.relu),  # (None,13,13,32)
            tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation=tf.nn.relu),  # (None,6,6,64)
            tf.keras.layers.Flatten(),                               # (None, 2304)
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),          # (None, 2*latent_dim) 输出 mu 和 sigma
            ]
        )
        self.inference_net.summary()

        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),    
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),  # (None,7*7*32)
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),    # (None,7,7,32)
            tf.keras.layers.Conv2DTranspose(         # (None,14,14,64)
                filters=64,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(         # (None,28,28,32)
                filters=32,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation=tf.nn.relu),
            # No activation
            tf.keras.layers.Conv2DTranspose(         # (None,28,28,1)
                filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
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


# ## Define the loss function and the optimizer
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
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    # 损失
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    print(tf.reduce_mean(cross_ent).numpy(), tf.reduce_mean(logqz_x-logpz).numpy())
    print((-tf.reduce_mean(logpx_z + logpz - logqz_x)).numpy())
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
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('vae-pic/image_at_epoch_{:04d}.png'.format(epoch))


generate_and_save_images(model, 0, random_vector_for_generation)

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for train_x in train_dataset:
        gradients, loss = compute_gradients(model, train_x)
        apply_gradients(optimizer, gradients, model.trainable_variables)
    end_time = time.time()

    if epoch % 1 == 0:
        loss = tfe.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = -loss.result()
        print('Epoch: {}, Test set ELBO: {}, ''time elapse for current epoch {}'.format(
                    epoch, elbo, end_time - start_time))
        generate_and_save_images(
            model, epoch, random_vector_for_generation)



# # ### Generate a GIF of all the saved images.

# # In[ ]:


# with imageio.get_writer('cvae.gif', mode='I') as writer:
#     filenames = glob.glob('image*.png')
#     filenames = sorted(filenames)
#     last = -1
#     for i,filename in enumerate(filenames):
#         frame = 2*(i**0.5)
#         if round(frame) > round(last):
#             last = frame
#         else:
#             continue
#         image = imageio.imread(filename)
#         writer.append_data(image)
#     image = imageio.imread(filename)
#     writer.append_data(image)
        
# # this is a hack to display the gif inside the notebook
# os.system('cp cvae.gif cvae.gif.png')


# # In[ ]:


# display.Image(filename="cvae.gif.png")


# To downlod the animation from Colab uncomment the code below:

# In[ ]:


#from google.colab import files
#files.download('cvae.gif')

