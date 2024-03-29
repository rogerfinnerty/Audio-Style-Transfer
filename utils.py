"""audio style transfer"""

import torch

def gram_matrix(input):
    """Compute gram matrix"""
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())  # inner product of
    return G.div(a * b * c * d)           # divide by layer dimension
