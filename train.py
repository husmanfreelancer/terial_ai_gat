import os

import torch
import json
from model import GATPredictor
from config import *

DATASET_PATH = "data/dataset.json"


def load_from_jsonl(path):
    """
    Loads and parses dataset.json containing:
    - nodes: list of job entries with features and labels
    - edges: list of node ID connections (source, target)

    Returns:
        x_tensor (Tensor): Input features (num_nodes x num_features)
        adj_matrix (Tensor): Graph adjacency matrix (num_nodes x num_nodes)
        y_tensor (Tensor): Output labels (num_nodes x 2)
    """
    with open(path, "r") as f:
        dataset = json.load(f)

    node_features = []
    labels = []
    id_to_index = {}

    for idx, node in enumerate(dataset["nodes"]):
        id_to_index[node["id"]] = idx
        f = node["features"]

        # Normalize and prepare features
        feature_vector = [
            f["num_workers"] / 10.0,
            f["material_cost"] / 10000.0,
            f["area_sqft"] / 5000.0,
            f["roof_pitch"] / 10.0
        ]
        node_features.append(feature_vector)

        l = node["label"]
        labels.append([
            l["estimated_cost"] / 10000.0,
            l["risk_score"]
        ])

    num_nodes = len(node_features)
    adj_matrix = torch.eye(num_nodes)

    for edge in dataset["edges"]:
        src = id_to_index[edge["source"]]
        tgt = id_to_index[edge["target"]]
        adj_matrix[src][tgt] = 1
        adj_matrix[tgt][src] = 1  # Undirected edge

    x_tensor = torch.tensor(node_features, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32)

    return x_tensor, adj_matrix, y_tensor


def train():
    x, adj, targets = load_from_jsonl(DATASET_PATH)

    model = GATPredictor(x.size(1), HIDDEN_DIM, OUTPUT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        outputs = model(x, adj)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} Loss: {loss.item():.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)


if __name__ == "__main__":
    train()
