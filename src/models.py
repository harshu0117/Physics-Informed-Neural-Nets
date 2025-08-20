# src/models.py
import torch
import torch.nn as nn
from typing import List

class PhysicsInformedNN(nn.Module):
    """
    A flexible Physics-Informed Neural Network class.
    
    Args:
        layers (List[int]): A list defining the network architecture. 
                            e.g., [2, 20, 20, 1] for 2 inputs, 2 hidden layers of 20 neurons, 1 output.
        lb (torch.Tensor): The lower bounds of the input domain.
        ub (torch.Tensor): The upper bounds of the input domain.
    """
    def __init__(self, layers: List[int], lb: torch.Tensor, ub: torch.Tensor):
        super(PhysicsInformedNN, self).__init__()
        
        self.layers = layers
        self.lb = lb
        self.ub = ub
        
        # Build the network
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.net.add_module(f"activation_{i}", nn.Tanh())
        
        # Initialize weights using Xavier Initialization, as in the original code
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the network.
        
        Args:
            x (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output of the network.
        """
        # Normalize the input to the range [-1, 1]
        x_normalized = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(x_normalized)