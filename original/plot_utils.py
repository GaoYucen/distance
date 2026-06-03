import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import networkx as nx

from utils.data_utils import print_green


## Function to set plot style
def set_plot_style(scale=1.5):
    """Set global plotting style parameters"""
    plt.style.use('default')  # Start with default style

    plt.rcParams.update({
        # Figure size and DPI
        'figure.figsize': (10*scale, 6*scale),
        'figure.dpi': 150,  # Screen display
        'savefig.dpi': 300,  # Saved figure resolution

        # Font sizes
        'font.size': 12*scale,  # Base font size
        'axes.titlesize': 14*scale,  # Title size
        'axes.labelsize': 12*scale,  # Axis label size
        'xtick.labelsize': 9*scale,  # X-axis tick label size
        'ytick.labelsize': 9*scale,  # Y-axis tick label size
        'legend.fontsize': 9*scale,  # Legend font size

        # Legend
        'legend.fontsize': 10*scale,
        'legend.title_fontsize': 12*scale,
    })

## Function to plot degree distribution
def plot_degree_distribution(G):
    degrees = dict(G.degree()).values()
    plt.hist(degrees, bins=50)
    plt.xlabel('Degree')
    plt.ylabel('Frequency')
    plt.title('Degree Distribution')
    plt.show()

def plot_data_distribution(data_train, data_test, data_name, query_name, dir_name, label=""):
    plt.figure(figsize=(10, 6))
    # Plot histogram of distances
    plt.hist(data_train, bins=100, alpha=0.5, label='Train', density=True)
    plt.hist(data_test, bins=100, alpha=0.5, label='Test', density=True)
    plt.xlabel("Distance [m]")
    plt.ylabel("Normalized Frequency (Density)")
    temp_label = f" - {label}" if label else ""
    plt.title(f"Histogram of Distances (Data: {data_name}{temp_label})")
    plt.legend()
    # Save or show the plot
    if 'get_ipython' in globals():
        plt.show()
    else:
        # Create the directory if it doesn't exist
        os.makedirs(dir_name, exist_ok=True)

        # Save the plot
        temp_label = f"_{label}" if label else ""
        file_name = os.path.join(dir_name, f"histogram_distances_{data_name}_{query_name}{temp_label}.png")
        print_green(f"Saving plot: {file_name}")
        plt.savefig(file_name)

## Function to plot the learning curves
def plot_learning_curves(history, n_batches, model_name, data_name, query_name, dir_name):
    if len(history['loss_epoch_history']) == 0:
        print("No training history available.")
        return

    n_epochs = len(history['loss_epoch_history'])

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate epochs and iterations
    epochs = (np.arange(0, n_epochs)*n_batches + np.arange(1, n_epochs+1)*n_batches)/2
    iterations = np.arange(1, len(history['loss_iter_history']) + 1)

    # Plot loss per epoch
    ax.plot(epochs, history['loss_epoch_history'], label='Loss per Epoch', color='blue', linewidth=2, marker='x', markersize=8)

    # Plot loss per iteration
    ax.plot(iterations, history['loss_iter_history'], label='Loss per Iteration', color='red', alpha=0.3)

    # Set labels and title for the primary axis
    ax.set_xlabel("Epochs / Iterations")
    ax.set_ylabel("Loss")
    ax.set_title(f"Learning Curves (Model: {model_name}, Data: {data_name})")
    ax.legend(loc="upper left")

    # Create a secondary y-axis for validation MRE
    if 'val_mre_epoch_history' in history and len(history['val_mre_epoch_history']) > 0:
        ax2 = ax.twinx()
        ax2.plot(epochs, history['val_mre_epoch_history'],
                 label='Validation MRE',
                 color='green', linewidth=2, marker='o', markersize=8,
                 zorder=1)
        ax2.set_ylabel("Validation MRE")
        ax2.legend(loc="upper right")
        # Set labels for the twin axis
        ax2.set_ylabel("Validation MRE")
        ax2.legend(loc="upper right")

    # Save or show the plot
    if 'get_ipython' in globals():
        plt.show()
    else:
        # Create the directory if it doesn't exist
        os.makedirs(dir_name, exist_ok=True)

        # Save the plot
        file_name = os.path.join(dir_name, f"loss_history_{model_name}_{data_name}_{query_name}.png")
        print_green(f"Saving plot: {file_name}")
        plt.savefig(file_name)

## Function to plot the targets (and their corresponding predictions) in a sorted order
def plot_targets_and_predictions(predictions, targets, model_name, data_name, query_name, dir_name, label=""):
    # Sort predictions and targets by the value of targets
    sorted_indices = np.argsort(targets)
    sorted_targets = targets[sorted_indices]
    sorted_predictions = predictions[sorted_indices]

    # Plot the sorted predictions and targets on a log scale
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_predictions, 'r.', markersize=1, label='Predictions')
    # plt.plot(np.abs(sorted_targets-sorted_predictions), 'r.', markersize=1, label='Absolute Error')
    plt.plot(sorted_targets, label='Targets')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance [m]')
    temp_label = f" - {label}" if label else ""
    plt.title(f"Predictions vs Targets (Model: {model_name}, Data: {data_name}{temp_label})")
    # plt.yscale('log')
    # plt.xscale('log')
    plt.legend()

    # Save or show the plot
    if 'get_ipython' in globals():
        plt.show()
    else:
        # Create the directory if it doesn't exist
        os.makedirs(dir_name, exist_ok=True)

        # Save the plot
        temp_label = f"_{label}" if label else ""
        file_name = os.path.join(dir_name, f"predictions_targets_{model_name}_{data_name}_{query_name}{temp_label}.png")
        print_green(f"Saving plot: {file_name}")
        plt.savefig(file_name)

def plot_targets_and_mre_boxplots(predictions, targets, model_name, data_name, query_name, dir_name, label="", num_buckets=5):
    """
    Plot MRE against targets using box plots for fixed ranges of target values.
    """
    mre = np.abs(predictions - targets) / targets  # Mean Relative Error

    # Bin targets into equal-sized ranges
    bucket_edges = np.linspace(targets.min(), targets.max(), num_buckets + 1, dtype=float)
    bucket_midpoints = []
    bucket_labels = []
    bucket_mre = []

    for i in range(num_buckets):
        start, end = bucket_edges[i], bucket_edges[i + 1]
        # Get the indices of targets that fall into the current bucket range
        bucket_indices = (targets >= start) & (targets < end)
        bucket_mre.append(100*mre[bucket_indices])  # Convert mre to percentage by multiplying by 100
        print(f"Bucket {i+1}: {start:.0f} - {end:.0f}, Local MRE: {bucket_mre[-1].mean():.2f}%, Fraction of data samples in this bucket: {bucket_indices.sum()/len(predictions)*100:.2f}%")
        bucket_labels.append(f"{start:.0f} - {end:.0f}")
        midpoint = (start + end) / 2
        bucket_midpoints.append(midpoint)
    print(f"Total Buckets: {len(bucket_mre)}, Global MRE: {100*mre.mean():.2f}%, Global Count: {len(predictions)}")

    # Calculate box plot width as a fraction of the bucket range
    box_width = (bucket_edges[1] - bucket_edges[0]) * 0.8

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot box plots
    ax.boxplot(bucket_mre, positions=bucket_midpoints, showfliers=False, showmeans=True,  widths=box_width)
    ax.set_xlabel("Target Buckets (Range) [m]")
    ax.set_ylabel("Mean Relative Error (MRE) [%]")
    temp_label = f" - {label}" if label else ""
    ax.set_title(f"MRE vs Targets (Model: {model_name}, Data: {data_name}{temp_label})")
    ax.tick_params(axis='x', rotation=0)

    # Set y-axis limits and grid lines for better readability
    ax.set_ylim(-0.5, 20)
    ax.set_yticks(np.arange(0, 21, 2))
    ax.yaxis.grid(True, which='major', linestyle='-', alpha=0.7)

    # Set x-axis ticks to the midpoints, rounded to 2 decimal places
    ax.set_xticks(bucket_midpoints)
    # ax.set_xticklabels([f"{midpoint:.2f}" for midpoint in bucket_midpoints], rotation=0)
    ax.set_xticklabels(bucket_labels, rotation=0)

    # Plot the distribution of targets on the same plot
    ax2 = ax.twinx()
    ax2.hist(targets, bins=50, color='gray', alpha=0.3, label='Target Distribution', density=True)
    # ax2.set_ylabel("Normalized Target Frequency")

    # Hide secondary axis ticks, labels and spines
    ax2.set_yticklabels([])  # Hide y-axis labels
    ax2.tick_params(right=False)  # Hide y-axis ticks
    ax2.spines['right'].set_visible(False)

    # Add custom legend for boxplot elements
    legend_elements = [
        Line2D([0], [0], color='orange', label='Median', linewidth=2),
        Line2D([0], [0], marker='^', color='w', label='Mean',
               markerfacecolor='green', markeredgecolor='green', markersize=8),
        Patch(facecolor='lightblue', edgecolor='black', label='IQR (25-75%)'),
        Line2D([0], [0], color='black', label='Whiskers\n(1.5xIQR)', linewidth=1)
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), title='Box Plot Elements')

    # Save or show the plot
    if 'get_ipython' in globals():
        plt.show()
    else:
        os.makedirs(dir_name, exist_ok=True)
        temp_label = f"_{label}" if label else ""
        file_name = os.path.join(dir_name, f"mre_boxplot_{model_name}_{data_name}_{query_name}{temp_label}.png")
        print_green(f"Saving plot: {file_name}")
        fig.savefig(file_name)

## Function to plot the ego graph of a node
def plot_subgraph(G, node=None, use_geo_coordinates=False, radius=3, node_size=300, with_labels=True, dpi=100):
    # Get the subgraph around the node
    if node:
        subgraph = nx.ego_graph(G, node, radius=radius)
    else:
        subgraph = G

    # Get the positions of the nodes
    if use_geo_coordinates is False:
        # Use spring layout if no coordinates are provided
        pos = nx.spring_layout(subgraph, seed=42)
    else:
        # Use the provided coordinates
        # TODO: this assumes that the node data has 'feature' attribute with coordinates as a tuple (x, y)
        pos = {node: (data['feature'][0], data['feature'][1]) \
                    for node, data in subgraph.nodes(data=True)}

    # Draw the subgraph
    plt.figure(figsize=(5,5), dpi=dpi)
    nx.draw(subgraph, pos, with_labels=with_labels,
            node_size=node_size, node_color='skyblue',
            edge_color='black', font_size=10)
    # Draw the given node in red
    if node:
        nx.draw_networkx_nodes(subgraph, pos, nodelist=[node], node_size=node_size, node_color='r')
        plt.title(f"{G.graph['data_name']}: Ego Graph of radius={radius} around node={node}")
    else:
        plt.title(f"{G.graph['data_name']}: Graph of {len(subgraph.nodes())} nodes")
    plt.show()
