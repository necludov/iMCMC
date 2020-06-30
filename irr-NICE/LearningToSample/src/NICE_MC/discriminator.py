""" Discriminator for the GAN training of a NICE sampler """
import tensorflow as tf
from tensorflow.layers import dense


class MLPDiscriminator:
    def __init__(self, arch, name='discriminator'):
        self.arch = arch
        self.name = name

    def __call__(self, x, reuse=True):
        with tf.variable_scope(self.name, reuse=reuse):
            for dim in self.arch:
                x = dense(x, dim, activation=tf.nn.leaky_relu)
            y = dense(x, 1)
        return y
