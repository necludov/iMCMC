""" Experiment to sample from a Mixture of Gaussians using NICE-MC"""
import argparse

import numpy as np

from LearningToSample.src.NICE_MC.nice_sampler import NiceSampler
from LearningToSample.src.experiment import Experiment
from LearningToSample.src.distributions import FeedForwardNetPost


def main(args):

    print('setting up distribution')
    def load_data(folder, test_frac=0.2):
        X = np.load(folder + '/data.npy')
        y = np.load(folder + '/labels.npy')
        N, D = X.shape
        data = np.concatenate([X, y], axis=1)
        np.random.shuffle(data)
        train_data = data[:int(N*(1-test_frac))]
        test_data = data[int(N*(1-test_frac)):]
        return train_data[:, :-1], train_data[:, -1], test_data[:, :-1], test_data[:, -1]

    X_train, y_train, X_test, y_test = load_data(args.data)
    data_dim = X_train.shape[1] + 1
    arch = [data_dim] + args.arch
    target_dim = np.sum([arch[i] * arch[i + 1] for i in range(len(arch) - 1)])
    gen_arch = 2 * [target_dim] + args.gen_arch
    dist = FeedForwardNetPost(X_train, y_train, X_test, y_test, arch, prec=args.prior)

    def noise(bs):
        return np.random.normal(0.0, 1.0, [bs, target_dim])

    print('setting up sampler')
    sampler = NiceSampler(gen_arch=gen_arch,
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
    parser.add_argument('--burn_in', type=int, default=5000)
    parser.add_argument('--logdir', type=str, default='logs/nice/net')
    parser.add_argument('--epoch_size', default=500, type=int,
                        help='number of iterations between updating bootstrap samples')
    parser.add_argument('--d_iters', default=5, type=int,
                        help="number of discriminator steps per generator step")
    parser.add_argument('--boostrap_steps', default=5000, type=int,
                        help='number of bootstrap samples to draw')
    parser.add_argument('--hmc_steps', default=50, type=int,
                        help='number of leap frog steps if hmc used')
    parser.add_argument('--hmc_step_size', default=0.001, type=float,
                        help='hmc step size to use during bootstrap')
    parser.add_argument('--b', default=8)
    parser.add_argument('--m', default=2)
    parser.add_argument('--bootstrap_burn_in', default=1000, type=int)
    parser.add_argument('--bootstrap_batch_size', default=32, type=int)
    parser.add_argument('-- bootstrap_discard_ratio', default=0.5, type=float)
    parser.add_argument('--evaluate_steps', default=5000, type=int)
    parser.add_argument('--evaluate_burn_in', default=1000, type=int)
    parser.add_argument('--evaluate_batch_size', default=32, type=int)
    parser.add_argument('--nice_steps', default=1, type=int)
    parser.add_argument('--use_hmc', default=True, type=bool)
    parser.add_argument('--hmc_epochs', default=5, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument("--max_iters", type=int, default=10000)
    parser.add_argument('--num_samples', type=int, default=25000,
                         help="Number of samples to draw")
    parser.add_argument('--gen_arch', type=list,
                        default=[[([400], 'v1', False),
                                        ([400], 'x1', True),
                                        ([400], 'v2', False)]])
    parser.add_argument('--disc_arch', type=list, default=[400, 400, 400])
    parser.add_argument('--debug', default=False, type=bool)
    parser.add_argument('--plot', default=False, type=bool)
    parser.add_argument('--data', default='data/boston', type=str,
                        help=" Data directory where data is stored")
    parser.add_argument('--prior', default=0.1, type=float,
                        help="the variance of the normal prior placed on"
                             "the logistic regression parameters")
    parser.add_argument('--arch', default=[50, 1], type=list,
                        help="layer widths for the bayesian net")
    args = parser.parse_args()
    main(args)