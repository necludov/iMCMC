""" A-NICE-MCMC Sampler"""
import time
import os

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from LearningToSample.src.sampler import Sampler
from LearningToSample.src.eval import batch_means_ess, acceptance_rate
from LearningToSample.src.NICE_MC.nice import create_nice_network
from LearningToSample.src.NICE_MC.nice import TrainingOperator
from LearningToSample.src.NICE_MC.nice import InferenceOperator, IrrInferenceOperator
from LearningToSample.src.NICE_MC.discriminator import MLPDiscriminator
from LearningToSample.src.NICE_MC.bootstrap import Buffer
from LearningToSample.src.HMC.hmc import HamiltonianMonteCarloSampler as HmcSampler


class NiceSampler(Sampler):

    def __init__(self, gen_arch, log_prob_func, disc_arch,
                 init_dist, gamma=1.0, scale=10.0, b=8, m=2):
        """ gen_arch:
                dictionary of parameters for initialising the generator
            log_prob_func:
                function to return the target log probability up till a constant
            disc_arch:
                dictionary of parameters needed to initialise the discriminator
            init_dist:
                the noise distribution used as input to the generator
            gamma:
                hyper-parameter used to maintain the closeness between the initial
                distribution on the auxiliary variable z and the learned distribution
            scale:
                size of the graident penalty used to enforce the 1-lipschitz constraint
                of the wassertein GAN objective
            b:
                Number of times we run the chain before drawing a second sample.
            m:
                Number of times we run the chain to get a second sample
                """

        self.energy_func = lambda var: -log_prob_func(var)
        self.generator = create_nice_network(*gen_arch)
        self.x_dim, self.v_dim = self.generator.x_dim, self.generator.v_dim
        self.discriminator = MLPDiscriminator(arch=disc_arch)
        self.train_op = TrainingOperator(self.generator)
        self.infer_op = IrrInferenceOperator(self.generator, self.energy_func)
        self.b = tf.to_int32(tf.reshape(tf.multinomial(tf.ones([1, b]), 1),
                                        [])) + 1
        self.m = tf.to_int32(
            tf.reshape(tf.multinomial(tf.ones([1, m]), 1), [])) + 1

        # Inputs to the pairwise discriminator
        self.z = tf.placeholder(tf.float32, [None, self.x_dim])
        self.x = tf.placeholder(tf.float32, [None, self.x_dim])
        self.xl = tf.placeholder(tf.float32, [None, self.x_dim])

        self.steps = tf.placeholder(tf.int32, [])
        self.nice_steps = tf.placeholder(tf.int32, [])
        self.hmc_sampler = None
        self.ns = init_dist
        self.ds = None
        bx, bz = tf.shape(self.x)[0], tf.shape(self.z)[0]

        # Obtain values from inference ops
        # `infer_op` contains Metropolis step
        v = tf.random_normal(tf.stack([bz, self.v_dim]))
        dir = tf.ones([bz, self.x_dim])
        self.z_, self.v_, self.dir_ = self.infer_op((self.z, v, dir), self.steps, self.nice_steps)

        # Reshape for pairwise discriminator
        x = tf.reshape(self.x, [-1, 2 * self.x_dim])
        xl = tf.reshape(self.xl, [-1, 2 * self.x_dim])

        # Obtain values from train ops
        v1 = tf.random_normal(tf.stack([bz, self.v_dim]))
        x1_, v1_ = self.train_op((self.z, v1), self.b)
        x1_ = x1_[-1]
        x1_sg = tf.stop_gradient(x1_)
        v2 = tf.random_normal(tf.stack([bx, self.v_dim]))
        x2_, v2_ = self.train_op((self.x, v2), self.m)
        x2_ = x2_[-1]
        v3 = tf.random_normal(tf.stack([bx, self.v_dim]))
        x3_, v3_ = self.train_op((x1_sg, v3), self.m)
        x3_ = x3_[-1]

        # The pairwise discriminator has two components:
        # (x, x2) from x -> x2
        # (x1, x3) from z -> x1 -> x3
        #
        # The optimal case is achieved when x1, x2, x3
        # are all from the data distribution
        x_ = tf.concat([
                tf.concat([x2_, self.x], 1),
                tf.concat([x3_, x1_], 1)
        ], 0)

        # Concat all v values for log-likelihood training
        v1_ = v1_[-1]
        v2_ = v2_[-1]
        v3_ = v3_[-1]
        v_ = tf.concat([v1_, v2_, v3_], 0)
        v_ = tf.reshape(v_, [-1, self.v_dim])

        d = self.discriminator(x, reuse=False)
        d_ = self.discriminator(x_)

        # generator loss

        # TODO: MMD loss (http://szhao.me/2017/06/10/a-tutorial-on-mmd-variational-autoencoders.html)
        # it is easy to implement, but maybe we should wait after this codebase is settled.
        self.v_loss = tf.reduce_mean(0.5 * tf.multiply(v_, v_))
        self.g_loss = tf.reduce_mean(d_) + self.v_loss * gamma

        # discriminator loss
        self.d_loss = tf.reduce_mean(d) - tf.reduce_mean(d_)

        epsilon = tf.random_uniform([], 0.0, 1.0)
        x_hat = xl * epsilon + x_ * (1 - epsilon)
        d_hat = self.discriminator(x_hat)
        ddx = tf.gradients(d_hat, x_hat)[0]
        ddx = tf.norm(ddx, axis=1)
        ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
        self.d_loss = self.d_loss + ddx

        # I don't have a good solution to the tf variable scope mess.
        # So I basically force the NiceLayer to contain the 'generator' scope.
        # See `nice/__init__.py`.
        g_vars = [var for var in tf.global_variables() if 'generator' in var.name]
        d_vars = [var for var in tf.global_variables() if self.discriminator.name in var.name]

        self.d_train = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5, beta2=0.9)\
            .minimize(self.d_loss, var_list=d_vars)
        self.g_train = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.5, beta2=0.9)\
            .minimize(self.g_loss, var_list=g_vars)

        self.init_op = tf.group(
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        )


    def sample(self, sess=None, num_samples=2000, nice_steps=1, num_chains=32, **kwargs):
        start = time.time()
        z, _ = sess.run([self.z_, self.v_], feed_dict={
            self.z: self.ns(num_chains), self.steps: num_samples,
            self.nice_steps: nice_steps})
        end = time.time()
        return z, end-start

    def sample_hmc(self, sess, steps=5000, batch_size=32, prior_scale=0.1,
                   hmc_step_size=0.001, hmc_steps=100):
        """ Returns samples from hmc """
        if not self.hmc_sampler:
            def prior(batch_size):
                s = np.random.normal(loc=0.0, scale=prior_scale, size=(batch_size, self.x_dim))
                return s
            self.hmc_sampler = HmcSampler(lambda x: - self.energy_func(x),
                                          prior,
                                          stepsize=hmc_step_size,
                                          n_steps=hmc_steps,
                                          data_dim=self.x_dim)
        return self.hmc_sampler.sample(sess, steps, batch_size)

    def bootstrap(self, sess, steps=5000, nice_steps=1, burn_in=1000, batch_size=32, discard_ratio=0.5,
                  use_hmc=False, **kwargs):
        if use_hmc:
            z, _ = self.sample_hmc(sess, steps + burn_in, batch_size, **kwargs)
        else:
            z, sample_time = self.sample(sess, steps + burn_in, nice_steps, batch_size)
        z = np.transpose(z, axes=[1, 0, 2])
        z = np.reshape(z[:, burn_in:], [-1, z.shape[-1]])
        if self.ds:
            self.ds.discard(ratio=discard_ratio)
            self.ds.insert(z)
        else:
            self.ds = Buffer(z)

    def train(self, sess=None,
              d_iters=5, epoch_size=1000, max_iters=100000,
              bootstrap_steps=5000, bootstrap_burn_in=1000,
              bootstrap_batch_size=32, bootstrap_discard_ratio=0.5,
              evaluate_steps=10000, evaluate_burn_in=5000, evaluate_batch_size=32, nice_steps=1,
              hmc_epochs=5, batch_size=32, use_hmc=False, save_path=None, log_freq=100,
              plot=True, save_freq=2000, hmc_steps=100, hmc_step_size=0.001,
              **kwargs):
        """
        Train the NICE proposal using adversarial training.
        :param d_iters: number of discrtiminator iterations for each generator iteration
        :param epoch_size: how many iteration for each bootstrap step
        :param log_freq: how many iterations for each log on screen
        :param max_iters: max number of iterations for training
        :param bootstrap_steps: how many steps for each bootstrap
        :param bootstrap_burn_in: how many burn in steps for each bootstrap
        :param bootstrap_batch_size: # of chains for each bootstrap
        :param bootstrap_discard_ratio: ratio for discarding previous samples
        :param evaluate_steps: how many steps to evaluate performance
        :param evaluate_burn_in: how many burn in steps to evaluate performance
        :param evaluate_batch_size: # of chains for evaluating performance
        :param nice_steps: Experimental.
            num of steps for running the nice proposal before MH. For now do not use larger than 1.
        :param hmc_epochs: how many bootstrap epochs to use HMC for before switiching to the model
        :return:
        """

        sess.run(self.init_op)
        saver = tf.train.Saver(max_to_keep=1)

        def _feed_dict(bs):
            return {self.z: self.ns(bs), self.x: self.ds(bs), self.xl: self.ds(4 * bs)}

        train_time = 0
        num_epochs = 0
        g_losses = []
        d_losses = []
        for t in range(0, max_iters):
            if num_epochs > hmc_epochs:
                use_hmc = False # eventually we want to stop using HMC as the bootstrap so we can improove upon it

            if t % epoch_size == 0:
                num_epochs += 1
                self.bootstrap(
                    sess=sess, steps=bootstrap_steps, burn_in=bootstrap_burn_in,
                    batch_size=bootstrap_batch_size, discard_ratio=bootstrap_discard_ratio,
                    use_hmc=use_hmc, hmc_steps=hmc_steps, hmc_step_size=hmc_step_size)

            if t % log_freq == 0:
                d_loss = sess.run(self.d_loss, feed_dict=_feed_dict(batch_size))
                g_loss, v_loss = sess.run([self.g_loss, self.v_loss], feed_dict=_feed_dict(batch_size))
                print('Iter [%d] time [%5.4f] d_loss [%.4f] g_loss [%.4f] v_loss [%.4f]' %
                                 (t, train_time, d_loss, g_loss, v_loss))
                g_losses.append(g_loss)
                d_losses.append(d_loss)

                if save_path:
                    z, sample_time = self.sample(sess,
                                                 evaluate_steps + evaluate_burn_in,
                                                 nice_steps,
                                                 evaluate_batch_size)
                    z = z[evaluate_burn_in:]
                    ess = batch_means_ess(z)
                    min_ess = np.mean(np.min(ess, axis=1), axis=0)
                    std_ess = np.std(np.min(ess, axis=1), axis=0)
                    acc_rate = acceptance_rate(z)
                    with open(
                            os.path.join(save_path['results'], 'results.txt'),
                            'a') as f:
                        f.write(
                            f"min ess at iteration {t}: {min_ess} +- {std_ess} \n")
                        f.write(f"acceptance at iteration {t}: {acc_rate} \n")

                    if plot:
                        print('plotting')

                        def plot2d(samples):
                            fig, ax = plt.subplots()
                            ax.hist2d(samples[:, 0, 0], samples[:, 0, 1],
                                      bins=400)
                            ax.set_aspect('equal', 'box')
                            plt.savefig(
                                os.path.join(save_path['figs'], 'samples.png'))
                            plt.close()

                        plot2d(z)

            if (t+1) % save_freq:
                saver.save(sess, os.path.join(save_path['ckpts'], 'ckpt'))

            start = time.time()
            for _ in range(0, d_iters):
                sess.run(self.d_train, feed_dict=_feed_dict(batch_size))
            sess.run(self.g_train, feed_dict=_feed_dict(batch_size))
            end = time.time()
            train_time += end - start

        g_losses = np.array(g_losses)
        d_losses = np.array(d_losses)
        losses = np.stack([g_losses, d_losses])

        return losses, train_time

