import time

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class HamiltonianMonteCarloSampler(object):
    """
    Thin wrapper around tfp HMC implementation to make running experiments easier.
    """
    def __init__(self, log_prob_fun, init_dist, stepsize=0.1, n_steps=10, data_dim=2):

        self.init_dist = init_dist
        self.current_state = tf.placeholder(dtype=tf.float32, shape=[None, data_dim])
        self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(log_prob_fun,
                                                         stepsize,
                                                         n_steps,
                                                         state_gradients_are_stopped=True)


    def sample(self, sess, num_samples, num_chains, **kwargs):
        samples, _ = tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=0,
            current_state=self.current_state,
            kernel=self.hmc_kernel
        )
        time1 = time.time()
        samples_ = sess.run(samples, feed_dict={self.current_state:
                                                self.init_dist(num_chains)})
        time2 = time.time()
        return samples_, time2 - time1


    def train(self, *args, **kwargs):
        return [0], [0]



def kinetic_energy(v):
    return 0.5 * tf.reduce_sum(tf.multiply(v, v), axis=1)


def hamiltonian(p, v, f):
    """
    Return the value of the Hamiltonian
    :param p: position variable
    :param v: velocity variable
    :param f: energy function
    :return: hamiltonian
    """
    return f(p) + kinetic_energy(v)
#
#
def metropolis_hastings_accept(energy_prev, energy_next):
    """
    Run Metropolis-Hastings algorithm for 1 step
    :param energy_prev:
    :param energy_next:
    :return: Tensor of boolean values, indicating accept or reject
    """
    energy_diff = energy_prev - energy_next
    return (tf.exp(energy_diff) - tf.random_uniform(tf.shape(energy_prev))) >= 0.0
