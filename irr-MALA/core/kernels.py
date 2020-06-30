import numpy as np
import torch
from core import iMCMC


def dirMH(target, dim, sigma, device):
    def swap_xv_flip_d(state):
        return {'x': state['v']+state['d'], 'v': state['v'], 'd': state['x']-state['v']}

    kernel1 = iMCMC.RWiMCMC(target, iMCMC.ProposalRW(sigma=sigma, dim=dim, device=device), swap_xv_flip_d)

    def kernel(state):
        state, accepted_mask = kernel1.iterate(state)
        state['d'] = -state['d']
        return state, accepted_mask

    return kernel


def rwMH(target, dim, sigma, device):
    def swap_xv(state):
        return {'x': state['v'].clone(), 'v': state['x'].clone()}

    kernel = iMCMC.RWiMCMC(target, iMCMC.ProposalRW(sigma=sigma, dim=dim, device=device), swap_xv)
    return lambda state: kernel.iterate(state)


def irrMALA(target, dim, step_size, sigma, device):
    def swap_xv_flip_d(proposal):
        def swap(state):
            grad_x = state['d'] * proposal.evaluate_grad(state['x'])
            grad_v = proposal.evaluate_grad(state['v'])
            sign = -torch.sign(torch.sum(grad_x * grad_v, dim=1)).reshape([-1, 1])
            return {'x': state['v'], 'v': state['x'], 'd': sign}

        return swap

    def flip_d(state):
        return {'x': state['x'], 'v': state['v'], 'd': -state['d'].clone()}

    proposal = iMCMC.ProposalMALA(target, step_size, sigma, dim, device)
    kernel0 = iMCMC.iMCMC(target, proposal, swap_xv_flip_d(proposal))
    kernel1 = flip_d

    def kernel(state):
        state, accepted_mask = kernel0.iterate(state)
        state = kernel1(state)
        return state, accepted_mask

    return kernel


def rwMALA(target, dim, step_size, sigma, device):
    def swap_xv(state):
        return {'x': state['v'], 'v': state['x'], 'd': state['d']}

    kernel0 = iMCMC.iMCMC(target, iMCMC.ProposalMALA(target, step_size, sigma, dim, device), swap_xv)

    def kernel(state):
        state, accepted_mask = kernel0.iterate(state)
        return state, accepted_mask

    return kernel
