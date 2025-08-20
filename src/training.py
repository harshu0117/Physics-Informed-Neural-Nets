# src/training.py

import torch
import time

class Trainer:
    """
    A class to handle the training of a PINN model.
    """
    def __init__(self, model, loss_fn, optimizer_Adam, optimizer_LBFGS):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_Adam = optimizer_Adam
        self.optimizer_LBFGS = optimizer_LBFGS
        self.loss_history = []

    def train(self, adam_epochs, lbfgs_epochs):
        """
        Trains the model using a hybrid Adam + L-BFGS approach.
        
        Args:
            adam_epochs (int): Number of epochs to train with Adam.
            lbfgs_epochs (int): Max number of iterations for L-BFGS.
        """
        print("--- Starting Adam Optimization ---")
        start_time = time.time()
        for epoch in range(adam_epochs):
            self.optimizer_Adam.zero_grad()
            loss = self.loss_fn()
            loss.backward()
            self.optimizer_Adam.step()
            self.loss_history.append(loss.item())
            
            if (epoch + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Adam Epoch: {epoch+1}/{adam_epochs}, Loss: {loss.item():.4e}, Time: {elapsed:.2f}s")
                start_time = time.time()
        
        if lbfgs_epochs > 0:
            print("\n--- Starting L-BFGS Optimization ---")
            
            # The L-BFGS optimizer in PyTorch requires a 'closure' function
            # that re-evaluates the model and returns the loss.
            def closure():
                self.optimizer_LBFGS.zero_grad()
                loss = self.loss_fn()
                loss.backward()
                self.loss_history.append(loss.item())
                return loss

            self.optimizer_LBFGS.step(closure)
            
            # The number of LBFGS iterations is not as straightforward as epochs.
            # We can log the final loss after the optimizer converges.
            final_lbfgs_loss = self.loss_fn()
            print(f"L-BFGS Final Loss: {final_lbfgs_loss.item():.4e}")