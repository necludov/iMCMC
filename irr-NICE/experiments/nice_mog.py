""" Experiment to sample from a Mixture of Gaussians using AVS"""
import argparse

import numpy as np

from LearningToSample.src.NICE_MC.nice_sampler import NiceSampler
from LearningToSample.src.experiment import Experiment
from LearningToSample.src.distributions import MixtureOfGaussians


def main(args):

    print('setting up distribution')
    dist = MixtureOfGaussians(means=args.means,
                              stds=args.stds,
                              pis=[0.5, 0.5])

    def noise(bs):
        return np.random.normal(0.0, 1.0, [bs, 2])

    print('setting up sampler')
    sampler = NiceSampler(gen_arch=args.gen_arch,
                          log_prob_func=dist.log_prob_func(),
                          disc_arch=args.disc_arch,
                          init_dist=noise,
                          b=args.b,
                          m=args.m)

    print('setting up and running experiment')
    exp = Experiment(log_dir=args.logdir,
                     sampler=sampler,
                     params=vars(args),
                     dist=dist,
                     debug=args.debug)

    exp.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--burn_in', type=int, default=500)
    parser.add_argument('-ld', '--logdir', type=str, default='logs/nice/mog')
    parser.add_argument('--epoch_size', default=500, type=int,
                        help='number of iterations between updating bootstrap samples')
    parser.add_argument('--d_iters', default=5, type=int,
                        help="number of discriminator steps per generator step")
    parser.add_argument('--boostrap_steps', default=5000, type=int,
                        help='number of bootstrap samples to draw')
    parser.add_argument('--b', default=8)
    parser.add_argument('--m', default=2)
    parser.add_argument('--bootstrap_burn_in', default=1000, type=int)
    parser.add_argument('--bootstrap_batch_size', default=32, type=int)
    parser.add_argument('-- bootstrap_discard_ratio', default=0.5, type=float)
    parser.add_argument('--evaluate_steps', default=5000, type=int)
    parser.add_argument('--evaluate_burn_in', default=1000, type=int)
    parser.add_argument('--evaluate_batch_size', default=32, type=int)
    parser.add_argument('--nice_steps', default=1, type=int)
    parser.add_argument('--use_hmc', default=False, type=bool)
    parser.add_argument('--hmc_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--max_iters', type=int, default=2000)
    parser.add_argument('--num_samples', type=int, default=20000,
                        help="Number of samples to draw")
    parser.add_argument('--num_chains', type=int, default=10)
    parser.add_argument('--gen_arch', type=list,
                        default=[2, 2, [([400], 'v1', False),
                                        ([400], 'x1', True),
                                        ([400], 'v2', False)]])
    parser.add_argument('--disc_arch', type=list, default=[400, 400, 400])
    parser.add_argument('--means', nargs='+', type=list,
                        default=[[10.0, 0.0], [-10.0, 0.0]], help='mog means')
    parser.add_argument('--stds', nargs='+', default=[[1.0, 1.0], [1.0, 1.0]],
                        type=list, help='mog stds')
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--plot', default=True, type=bool)
    args = parser.parse_args()
    main(args)