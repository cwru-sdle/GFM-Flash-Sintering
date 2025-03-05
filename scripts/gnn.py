import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import Sequential, GCNConv
from torch_geometric.utils import to_undirected
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import joblib
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

'''
GNN Scripts for Flash Sintering Foundation Model approach
'''

# Simple GNN model with 3 GCN layers and a final fully connected layer
# Only uses for proof of concept, we transition to more capable and complex ML pipeline (eg. ALIGNN, stGNN)
class KneePointGCN(nn.Module):
    def __init__(self, features=10, output_channels=2):
        super(KneePointGCN, self).__init__()
        self.conv1 = GCNConv(features, 16)
        self.conv2 = GCNConv(16, 16)
        self.conv3 = GCNConv(16, 16)
        self.final_layer = nn.Linear(16, output_channels)

    def forward(self, x, edge_index):
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = self.final_layer(x)
        return x

# Visualize graph construction
def visualize_graph(graph_data, title="Graph Visualization"):
    """
    Function to visualize a PyG graph using NetworkX.
    """
    # Convert pytorch geometrics graph to NetworkX format
    G = to_networkx(graph_data, to_undirected=True)
    print(G.nodes)
    # Set figure size
    plt.figure(figsize=(15, 5))

    pos = nx.spring_layout(G, seed=42, k=0.4)
    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=200, font_size=8)

    # Set title
    plt.title(title)
    plt.show()

# Train model function
def train(model, train_data, val_data, test_data, optimizer, loss_fn, epochs):
    history = {
        'train_losses': [],
        'test_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'test_accuracies': [],
        'val_accuracies': []
    }
    for epoch in range(epochs):
        # Begin training
        model.train()
        optimizer.zero_grad()
        output = model(train_data.x, train_data.edge_index)

        percentage_error = torch.abs((output - train_data.y) / (train_data.y + 1e-8)) * 100
        mape = percentage_error.mean().item()
        epoch_accuracy = 100 - mape
        history['train_accuracies'].append(epoch_accuracy)
        loss = loss_fn(output, train_data.y)
        history['train_losses'].append(loss.item())
     
        loss.backward()
        optimizer.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item()}, Train Accuracy: {epoch_accuracy}%')

        # Begin Validation
        model.eval()
        with torch.inference_mode():
            val_output = model(val_data.x, val_data.edge_index)
            percentage_error = torch.abs((val_output - val_data.y) / (val_data.y + 1e-8)) * 100
            val_mape = percentage_error.mean().item()
            val_accuracy = 100 - val_mape
            history['val_accuracies'].append(val_accuracy)
            val_loss = loss_fn(val_output, val_data.y)
            history['val_losses'].append(val_loss.item())
        print(f'Val Loss: {val_loss.item()}, Val Accuracy: {val_accuracy}%')
    
    # Begin Testing
    model.eval()
    with torch.inference_mode():
        test_output = model(test_data.x, test_data.edge_index)

        percentage_error = torch.abs((test_output - test_data.y) / (test_data.y + 1e-8)) * 100
        test_mape = percentage_error.mean().item()
        test_accuracy = 100 - test_mape
        history['test_accuracies'].append(test_accuracy)
        test_loss = loss_fn(test_output, test_data.y)
        history['test_losses'].append(test_loss.item())
    print(f'Test Loss: {test_loss.item()}, Test Accuracy: {test_accuracy}%')
    return history

# Plot loss functions and training and validation accuracy throughout epochs iterations
def plot_losses(epochs, training_losses, training_accuracies, test_accuracies, test_losses, val_losses, val_accuracies, save_path):  
    fig, ax = plt.subplots(2, 1, figsize=(40, 20))
    ax[0].plot(range(epochs), training_losses, label='Training Loss', color='blue')
    ax[0].plot(range(epochs), val_losses, label='Validation Loss', color='green')
    ax[0].legend()
    ax[1].plot(range(epochs), training_accuracies, label='Training Accuracy', color='blue')
    ax[1].plot(range(epochs), val_accuracies, label='Validation Accuracy', color='green')
    ax[1].legend()
    if save_path is not None:
        plt.savefig(save_path + '/loss_figure.png')
    else:
        plt.show()