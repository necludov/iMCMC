import numpy as np
import torch
import torch.nn.functional as F

from torch import distributions


class MOGSix:
    def __init__(self, device):
        self.num_dims = 2
        cov_matrix = torch.from_numpy(0.5 * np.eye(2)).to(device).float()
        self.mean1 = torch.Tensor([-5, 0]).to(device).float()
        self.mean2 = torch.Tensor([5, 0]).to(device).float()
        self.mean3 = torch.Tensor([-2.5, 2.5 * np.sqrt(3)]).to(device).float()
        self.mean4 = torch.Tensor([2.5, -2.5 * np.sqrt(3)]).to(device).float()
        self.mean5 = torch.Tensor([-2.5, -2.5 * np.sqrt(3)]).to(device).float()
        self.mean6 = torch.Tensor([2.5, 2.5 * np.sqrt(3)]).to(device).float()

    def log_prob(self, x):
        prob1 = torch.exp(-2*torch.sum((x - self.mean1) ** 2, dim=1)) / np.sqrt(2 * np.pi * 0.25)
        prob2 = torch.exp(-2*torch.sum((x - self.mean2) ** 2, dim=1)) / np.sqrt(2 * np.pi * 0.25)
        prob3 = torch.exp(-2*torch.sum((x - self.mean3) ** 2, dim=1)) / np.sqrt(2 * np.pi * 0.25)
        prob4 = torch.exp(-2*torch.sum((x - self.mean4) ** 2, dim=1)) / np.sqrt(2 * np.pi * 0.25)
        prob5 = torch.exp(-2*torch.sum((x - self.mean5) ** 2, dim=1)) / np.sqrt(2 * np.pi * 0.25)
        prob6 = torch.exp(-2*torch.sum((x - self.mean6) ** 2, dim=1)) / np.sqrt(2 * np.pi * 0.25)
        return torch.log(prob1 + prob2 + prob3 + prob4 + prob5 + prob6 + 1e-10) - np.log(6)

    @staticmethod
    def mean():
        return np.array([0.0, 0.0])

    @staticmethod
    def std():
        return np.array([3.57, 3.57])

    @staticmethod
    def statistics(z):
        return z

    @staticmethod
    def xlim():
        return [-8, 8]

    @staticmethod
    def ylim():
        return [-8, 8]


class MOGTwo:
    def __init__(self, device):
        self.num_dims = 2
        cov_matrix = torch.from_numpy(0.5 * np.eye(2)).to(device).float()
        self.mean1 = torch.Tensor([-2, 0]).to(device).float()
        self.mean2 = torch.Tensor([2, 0]).to(device).float()

    def log_prob(self, state):
        x = state['x']
        prob1 = torch.exp(-torch.sum((x - self.mean1) ** 2, dim=1))
        prob2 = torch.exp(-torch.sum((x - self.mean2) ** 2, dim=1))
        return torch.log(prob1 + prob2 + 1e-10) - np.log(2)

    @staticmethod
    def mean():
        return np.array([0.0, 0.0])

    @staticmethod
    def std():
        return np.sqrt(np.array([4.5, 0.5]))

    @staticmethod
    def statistics(z):
        return z

    @staticmethod
    def xlim():
        return [-8, 8]

    @staticmethod
    def ylim():
        return [-8, 8]


class MOG:
    def __init__(self, means, cov_matrix, device):
        self.num_dims = 2
        self.g = []
        cov_matrix = torch.from_numpy(cov_matrix).to(device).float()
        for mean in means:
            mean = torch.Tensor(mean).to(device).float()
            self.g.append(distributions.multivariate_normal.MultivariateNormal(mean, cov_matrix))

    def log_prob(self, x):
        prob = 0.0
        for g in self.g:
            prob += torch.exp(g.log_prob(x)) / len(self.g)
        return torch.log(prob + 1e-10)


class Ring:
    def __init__(self, device):
        self.num_dims = 2
        self.device = device

    def energy(self, x):
        assert x.shape[1] == 2
        return (torch.sqrt(torch.sum(x ** 2, dim=1)) - 2.0) ** 2 / 0.32

    def log_prob(self, x):
        return -self.energy(x)

    @staticmethod
    def mean():
        return np.array([0., 0.])

    @staticmethod
    def std():
        return np.array([1.497, 1.497])

    @staticmethod
    def statistics(z):
        return z

    @staticmethod
    def xlim():
        return [-4, 4]

    @staticmethod
    def ylim():
        return [-4, 4]


class Rings:
    def __init__(self, device):
        self.num_dims = 2
        self.device = device

    def energy(self, x):
        assert x.shape[1] == 2
        p1 = (torch.sqrt(torch.sum(x ** 2, dim=1)) - 1.0) ** 2 / 0.04
        p2 = (torch.sqrt(torch.sum(x ** 2, dim=1)) - 2.0) ** 2 / 0.04
        p3 = (torch.sqrt(torch.sum(x ** 2, dim=1)) - 3.0) ** 2 / 0.04
        p4 = (torch.sqrt(torch.sum(x ** 2, dim=1)) - 4.0) ** 2 / 0.04
        p5 = (torch.sqrt(torch.sum(x ** 2, dim=1)) - 5.0) ** 2 / 0.04
        return torch.min(torch.min(torch.min(torch.min(p1, p2), p3), p4), p5)

    def log_prob(self, x):
        return -self.energy(x)

    @staticmethod
    def mean():
        return np.array([3.6])

    @staticmethod
    def std():
        return np.array([1.24])

    @staticmethod
    def xlim():
        return [-6, 6]

    @staticmethod
    def ylim():
        return [-6, 6]

    @staticmethod
    def statistics(z):
        z_ = torch.sqrt(torch.sum(z**2, dim=1, keepdims=True))
        return z_


class BayesianLogisticRegression:
    def __init__(self, data, labels, device):
        self.data = torch.tensor(data).to(device).float()
        self.labels = torch.tensor(labels).to(device).float().flatten()
        self.num_features = self.data.shape[1]
        self.num_dims = self.num_features + 1

    def view_params(self, v):
        w = v[:,:self.num_features].view([-1, self.num_features, 1])
        b = v[:,self.num_features:].view([-1, 1, 1])
        return w, b

    def energy(self, v):
        w, b = self.view_params(v)
        x = self.data
        y = self.labels.view([1,-1,1])
        logits = torch.matmul(x,w) + b
        probs = torch.nn.Sigmoid()(logits)
        nll = -y*torch.log(probs + 1e-16) - (1.0-y)*torch.log(1.0-probs  + 1e-16)
        nll = torch.sum(nll, dim=[1,2])
        negative_logprior = torch.sum(0.5*w**2/0.1, dim=[1,2])
        return negative_logprior + nll

    def log_prob(self, v):
        return -self.energy(v['x'])


class Australian(BayesianLogisticRegression):
    def __init__(self, device):
        data = np.load('../data/australian/data.npy')
        labels = np.load('../data/australian/labels.npy')

        dm = np.mean(data, axis=0)
        ds = np.std(data, axis=0)
        data = (data - dm) / ds

        super(Australian, self).__init__(data, labels, device)

    @staticmethod
    def mean():
        return np.array([
            0.00573914,  0.01986144, -0.15868089,  0.36768475,  0.72598995,  0.08102263,
            0.25611847,  1.68464095,  0.19636668,  0.65685423, -0.14652498,  0.15565136,
            -0.32924402,  1.6396836,  -0.31129081])

    @staticmethod
    def std():
        return np.array([
            0.12749956,  0.13707998,  0.13329148,  0.12998348,  0.14871537,  0.14387384,
            0.16227234,  0.14832425,  0.16567627,  0.26399282,  0.12827283,  0.12381153,
            0.14707848,  0.56716324,  0.15749387])

    @staticmethod
    def statistics(z):
        return z


class German(BayesianLogisticRegression):
    def __init__(self, device):
        data = np.load('../data/german/data.npy')
        labels = np.load('../data/german/labels.npy')

        dm = np.mean(data, axis=0)
        ds = np.std(data, axis=0)
        data = (data - dm) / ds

        super(German, self).__init__(data, labels, device)

    @staticmethod
    def mean():
        return np.array([
            -0.73619639,  0.419458, -0.41486377,  0.12679717, -0.36520298, -0.1790139,
            -0.15307771,  0.01321516,  0.18079792, - 0.11101034, - 0.22463548,  0.12258933,
            0.02874339, -0.13638893, -0.29289896,  0.27896283, -0.29997425,  0.30485174,
            0.27133239,  0.12250612, -0.06301813, -0.09286941, -0.02542205, -0.02292937,
            -1.20507437])

    @staticmethod
    def std():
        return np.array([
            0.09370191,  0.1066482,   0.097784,    0.11055009,  0.09572253,  0.09415687,
            0.08297686,  0.0928196,   0.10530122,  0.09953667,  0.07978824,  0.09610339,
            0.0867488,   0.09550436,  0.11943925,  0.08431934,  0.10446487,  0.12292658,
            0.11318609,  0.14040756,  0.1456459,   0.09062331,  0.13020753,  0.12852231,
            0.09891565])

    @staticmethod
    def statistics(z):
        return z


class Heart(BayesianLogisticRegression):
    def __init__(self, device):
        data = np.load('../data/heart/data.npy')
        labels = np.load('../data/heart/labels.npy')

        dm = np.mean(data, axis=0)
        ds = np.std(data, axis=0)
        data = (data - dm) / ds

        super(Heart, self).__init__(data, labels, device)

    @staticmethod
    def mean():
        return np.array([
            -0.13996868,  0.71390106,  0.69571619,  0.43944853,  0.36997702, -0.27319424,
            0.31730518, -0.49617367,  0.40516419, 0.4312388,   0.26531786, 1.10337417,
            0.70054367, -0.25684964])

    @staticmethod
    def std():
        return np.array([
            0.22915648,  0.24545612,  0.20457998,  0.20270157,  0.21040644,  0.20094482,
            0.19749419,  0.24134014,  0.20230987,  0.25595334,  0.23709087,  0.24735325,
            0.20701178,  0.19771984])

    @staticmethod
    def statistics(z):
        return z


class ICG:
    def __init__(self, device):
        self.num_dims = 50
        self.variances = torch.from_numpy(10**np.linspace(-2.0, 2.0, self.num_dims)).to(device).float()

    def log_prob(self, x):
        assert x.shape[1] == self.num_dims
        return -0.5*torch.sum(x*(x*1/self.variances), dim=1)

    def mean(self):
        return np.zeros(self.num_dims)

    def std(self):
        return np.sqrt(10**np.linspace(-2.0, 2.0, self.num_dims))


class SCG:
    def __init__(self, device):
        self.num_dims = 2
        self.variances = torch.from_numpy(10 ** np.linspace(-2.0, 2.0, self.num_dims)).to(device).float()
        B = torch.from_numpy(np.array([[1 / np.sqrt(2), -1 / np.sqrt(2)],
                                       [1 / np.sqrt(2), 1 / np.sqrt(2)]])).float().to(device)
        self.cov_matrix = B.mm(torch.diag(self.variances).mm(B.t()))
        self.inv_cov = torch.inverse(self.cov_matrix)

    def log_prob(self, x):
        assert x.shape[1] == self.num_dims
        return -0.5*torch.sum(x*self.inv_cov.mm(x.t()).t(), dim=1)

    def mean(self):
        return np.zeros(2)

    def std(self):
        return torch.sqrt(torch.diag(self.cov_matrix)).cpu().numpy()


class L2HMC_MOGTwo:
    def __init__(self, device):
        self.num_dims = 2
        self.variance = 0.1
        self.mean1 = torch.Tensor([-2, 0]).to(device).float()
        self.mean2 = torch.Tensor([2, 0]).to(device).float()

    def log_prob(self, x):
        assert x.shape[1] == self.num_dims
        prob1 = torch.exp(-0.5/self.variance*torch.sum((x - self.mean1) ** 2, dim=1))
        prob2 = torch.exp(-0.5/self.variance*torch.sum((x - self.mean2) ** 2, dim=1))
        return torch.log(prob1 + prob2 + 1e-10) - np.log(2.0)

    @staticmethod
    def mean():
        return np.array([0.0, 0.0])

    def std(self):
        return np.array([np.sqrt(4.1), np.sqrt(self.variance)])


class RoughWell:
    def __init__(self, device):
        self.num_dims = 2
        self.eta = 1e-2

    def log_prob(self, x):
        assert x.shape[1] == self.num_dims
        return -torch.sum(0.5 * x*x + self.eta*torch.cos(x/self.eta), dim=1)

    @staticmethod
    def mean():
        return np.array([0.0, 0.0])

    def std(self):
        return np.ones(self.num_dims)
