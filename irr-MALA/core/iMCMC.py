import numpy as np
import torch


class Normal:
    def __init__(self, mu, sigma, dim):
        self.loc = mu
        self.normal = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu,
                                                                                 covariance_matrix=sigma)

    def log_prob(self, state):
        assert ('x' in state.keys())
        return self.normal.log_prob(state['x'])


class AugmentedTarget:
    '''p(x)UniformSphere(d|0,alpha)'''

    def __init__(self, target, dim):
        self.target = target
        self.dim = dim

    def log_prob(self, state):
        return self.target.log_prob(state)


class ProposalRW:
    '''N(v|x,diag(sigma^2))'''

    def __init__(self, sigma, dim, device):
        self.sigma = sigma
        self.dim = dim
        self.device = device

    def sample(self, state):
        return self._sample(state['x'])

    def _sample(self, x):
        assert (x.shape[1] == self.dim)
        return x + self.sigma * torch.empty(x.size()).normal_().to(self.device)


class ProposalX:
    '''N(v|x+d,diag(sigma^2))'''

    def __init__(self, sigma, dim, device):
        self.sigma = sigma
        self.dim = dim
        self.device = device

    def log_prob(self, state):
        return self._log_prob(self, state['v'], state['x'], state['d'])

    def _log_prob(self, v, x, d):
        assert (x.shape[1] == self.dim) and (d.shape[1] == self.dim) and (v.shape[1] == self.dim)
        mu = x + d
        return -0.5 * torch.sum((v - mu) ** 2, dim=1) / self.sigma ** 2 - 0.5 * self.dim * np.log(
            2 * np.pi * self.sigma ** 2)

    def sample(self, state):
        return self._sample(state['x'], state['d'])

    def _sample(self, x, d):
        assert (x.shape[1] == self.dim) and (d.shape[1] == self.dim)
        return x + d + self.sigma * torch.empty_like(x).normal_()


class ProposalD:
    def __init__(self, alpha, sigma, dim, device):
        self.alpha = alpha
        self.sigma = sigma
        self.dim = dim
        self.device = device
        self.i = 0

    def sample(self, state):
        return self._sample(state['d'])

    def _sample(self, d):
        if self.i % 100 == 0:
            assert (d.shape[1] == self.dim)
            samples = self.sigma * torch.empty_like(d).normal_()
            samples = self.alpha * samples / torch.norm(samples, dim=1).reshape([-1, 1])
        else:
            samples = d
        self.i += 1
        return samples


class ProposalMALA:
    def __init__(self, target, step_size, sigma, dim, device):
        self.target = target
        self.step_size = step_size
        self.sigma = sigma
        self.dim = dim
        self.device = device

    def log_prob(self, state):
        assert ('x' in state.keys()) and ('v' in state.keys()) and ('d' in state.keys())
        return self._log_prob(state['x'], state['v'], state['d'])

    def _log_prob(self, x, v, d):
        assert (x.shape[1] == self.dim) and (d.shape[1] == 1) and (v.shape[1] == self.dim)
        grad_x = self.evaluate_grad(x)
        mean = x + self.step_size * d * grad_x
        log_p = -0.5 * torch.sum((v - mean) ** 2, dim=1) / self.sigma ** 2 - 0.5 * self.dim * np.log(
            2 * np.pi * self.sigma ** 2)
        return log_p

    def sample(self, state):
        assert ('x' in state.keys()) and ('d' in state.keys())
        return self._sample(state['x'], state['d'])

    def _sample(self, x, d):
        grad_x = self.evaluate_grad(x)
        samples = x + self.step_size * d * grad_x + self.sigma * torch.empty(x.size()).normal_().to(self.device)
        return samples.detach()

    def evaluate_grad(self, x):
        y = x.detach()
        y.requires_grad = True
        grad = torch.autograd.grad(self.target.log_prob({'x': y}), y, torch.ones([y.shape[0]]).to(self.device))[0]
        assert (grad.shape[0] == x.shape[0]) and (grad.shape[1] == self.dim)
        return grad.detach()


class AAiMCMC:
    def __init__(self, target, aux, f):
        self.target = target
        self.aux = aux
        self.f = f

    def iterate(self, current_states):
        current_states['v'] = self.aux.sample(current_states)
        next_states = self.f(current_states)
        return next_states


class RWiMCMC:
    def __init__(self, target, aux, f):
        self.target = target
        self.aux = aux
        self.f = f

    def iterate(self, current_states):
        current_states['v'] = self.aux.sample(current_states)
        current_log_p = self.target.log_prob(current_states)
        next_states = self.f(current_states)
        next_log_p = self.target.log_prob(next_states)
        test = (next_log_p - current_log_p).flatten()
        u = torch.ones(len(test)).uniform_().to(test.device)
        accepted_mask = test > torch.log(u)
        for k in current_states.keys():
            next_states[k][1 - accepted_mask] = current_states[k][1 - accepted_mask]
        return next_states, accepted_mask


class iMCMC:
    def __init__(self, target, aux, f):
        self.target = target
        self.aux = aux
        self.f = f

    def iterate(self, current_states):
        current_states['v'] = self.aux.sample(current_states)
        current_log_p = self.target.log_prob(current_states) + self.aux.log_prob(current_states)
        next_states = self.f(current_states)
        next_log_p = self.target.log_prob(next_states) + self.aux.log_prob(next_states)
        test = (next_log_p - current_log_p).flatten()
        u = torch.ones(len(test)).uniform_().to(test.device)
        accepted_mask = test > torch.log(u)
        for k in current_states.keys():
            next_states[k][1 - accepted_mask] = current_states[k][1 - accepted_mask]
        next_log_p[1 - accepted_mask] = current_log_p[1 - accepted_mask]
        return next_states, accepted_mask
