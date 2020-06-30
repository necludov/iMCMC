""" Very light base class for all the samplers"""


class Sampler:

    def train(self, sess, **kwargs):
        raise NotImplementedError

    def sample(self, sess, num_samples, **kwargs):
        raise NotImplementedError