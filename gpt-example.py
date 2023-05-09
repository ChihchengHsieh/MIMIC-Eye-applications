import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# Define the nodes and edges of the graph
# Here, we assume that we have an array of eye positions `positions`
# `threshold` is the distance threshold for connecting fixations with an edge
positions = np.array([[100, 200], [150, 250], [200, 300], [250, 350], [300, 400]])
threshold = 50

nodes = []
edges = []

# Create nodes
for i, pos in enumerate(positions):
    node = {
        "x": pos[0],
        "y": pos[1],
        "fixation_duration": 1.0, # Example feature
        "time_between_fixations": 1.0 # Example feature
    }
    nodes.append(node)

# Create edges
for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        if i != j:
            distance = np.sqrt((node1["x"] - node2["x"]) ** 2 + (node1["y"] - node2["y"]) ** 2)
            if distance <= threshold:
                edges.append((i, j))

# Create feature vectors and adjacency matrix
x = torch.tensor([list(node.values()) for node in nodes], dtype=torch.float)
edge_index = torch.tensor(list(zip(*edges)), dtype=torch.long)

# Create the data object for PyTorch Geometric
data = Data(x=x, edge_index=edge_index)

# Define the Graph Neural Network model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(x.size()[1], 16)
        self.conv2 = GCNConv(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x.view(-1)

# Train the model
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, torch.tensor([1.0])) # Example target value
    loss.backward()
    optimizer.step()

# Make predictions
prediction = model(data)

print(prediction)


# https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html#torch_geometric.nn.conv.GCNConv