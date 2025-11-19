import torch

def get_available_device():
    """Get available GPU device."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if torch.mps.is_available():
        return torch.device('mps:0')
    return torch.device('cpu')
