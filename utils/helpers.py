"""Helper Functions"""
import torch
import numpy as np

def set_random_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device(use_cuda=True):
    return torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
