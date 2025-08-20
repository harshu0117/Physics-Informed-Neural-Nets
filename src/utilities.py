# src/utilities.py
import scipy.io
import torch
import numpy as np
import random

def load_data(file_path):
    """
    Loads data from a .mat file.
    
    Args:
        file_path (str): The path to the .mat file.
        
    Returns:
        tuple: A tuple containing the loaded data arrays (e.g., x, t, u).
               The exact contents depend on the .mat file structure.
    """
    try:
        data = scipy.io.loadmat(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return None

def relative_l2_error(pred_tensor, true_tensor):
    """
    Calculates the relative L2 error between two tensors.
    
    Args:
        pred_tensor (torch.Tensor): The predicted tensor.
        true_tensor (torch.Tensor): The ground truth tensor.
        
    Returns:
        float: The relative L2 error.
    """
    pred = pred_tensor.detach().cpu().numpy()
    true = true_tensor.detach().cpu().numpy()
    
    error_norm = np.linalg.norm(true - pred, 2)
    true_norm = np.linalg.norm(true, 2)
    
    return error_norm / true_norm

def setup_seed(seed):
    """
    Sets the seed for reproducibility.
    
    Args:
        seed (int): The seed to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True