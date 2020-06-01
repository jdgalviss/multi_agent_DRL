import torch
import numpy as np
import copy
import random

class OUNoise(object):
    """Ornstein-Uhlenbeck process"""
    
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """ initialise noise parameters """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = 1e-2
        self.seed = torch.manual_seed(seed)
        self.reset()
        
    def reset(self):
        """reset the internal state to mean (mu)"""
        self.state = copy.copy(self.mu)
        
    def sample(self):
        """Update internal state and return it as a noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.array([np.random.normal() for i in range(len(x))])
        self.state = x + dx
        return self.state

def hard_update(local_model, target_model):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
            
            
def soft_update(local_model, target_model, tau):
        """
            Soft update model parameters.
            θ_target = τ*θ_local + (1 - τ)*θ_target
            Params
            ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)