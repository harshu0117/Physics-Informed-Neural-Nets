# src/visualisation.py

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def newfig(width, height):
    """
    Helper function to create a new figure with specific dimensions.
    This is from the author's original utilities.
    """
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(111)
    return fig, ax

def plot_solution_heatmap(x, t, u, title="Solution", cmap='rainbow'):
    """
    Plots a 2D heatmap of the solution u(t, x).
    
    Args:
        x (np.array): Spatial coordinates (1D).
        t (np.array): Temporal coordinates (1D).
        u (np.array): Solution grid (T x X).
        title (str): The title of the plot.
        cmap (str): Colormap for the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    T, X = np.meshgrid(t, x)
    h = ax.pcolormesh(T, X, u.T, cmap=cmap, shading='gouraud')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_snapshots(x, t, u_exact, u_pred, snapshot_times):
    """
    Plots snapshots of the solution at specific time points.
    
    Args:
        x (np.array): Spatial coordinates (1D).
        t (np.array): Temporal coordinates (1D).
        u_exact (np.array): The exact solution grid (T x X).
        u_pred (np.array): The predicted solution grid (T x X).
        snapshot_times (list): A list of time points to plot.
    """
    fig, axes = plt.subplots(1, len(snapshot_times), figsize=(5 * len(snapshot_times), 5), sharey=True)
    
    for i, t_snap in enumerate(snapshot_times):
        ax = axes[i]
        t_idx = np.argmin(np.abs(t - t_snap))
        
        ax.plot(x, u_exact[t_idx, :], 'b-', linewidth=2, label='Exact')
        ax.plot(x, u_pred[t_idx, :], 'r--', linewidth=2, label='Prediction')
        ax.set_title(f'$t = {t[t_idx][0]:.2f}$')
        ax.set_xlabel('$x$')
        if i == 0:
            ax.set_ylabel('$u(t,x)$')
        ax.legend()
        ax.grid(True)
        
    plt.tight_layout()
    plt.show()

def plot_loss_history(history, title="Loss History"):
    """
    Plots the loss history during training.
    
    Args:
        history (list): A list of loss values.
        title (str): The title of the plot.
    """

    plt.figure(figsize=(8, 6))
    plt.plot(history)
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
