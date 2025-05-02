import torch
import json
from model import GATPredictor
from config import MODEL_PATH, HIDDEN_DIM, OUTPUT_DIM

DATASET_PATH = "data/real-time-predict.jsonl"


def load_existing_data():
    """
    Loads previously known project/job entries from a JSONL file.
    Each entry includes structured features consistent with the training data.
    Returns feature tensor and placeholder adjacency (self-loop) tensor.
    """
    x_data = []
    with open(DATASET_PATH, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                feats = item["features"]
                x_data.append([
                    feats["num_workers"] / 10.0,
                    feats["material_cost"] / 10000.0,
                    feats["area_sqft"] / 5000.0,
                    feats["roof_pitch"] / 10.0
                ])
            except (KeyError, json.JSONDecodeError) as e:
                print(f"Skipping line due to error: {e}")
                continue

    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    adj = torch.eye(len(x_tensor))
    return x_tensor, adj


def encode_new_node(record):
    """
    Converts a new job/project record into a scaled tensor, matching the training data scaling.
    """
    return torch.tensor([[
        record["features"]["num_workers"] / 10.0,
        record["features"]["material_cost"] / 10000.0,
        record["features"]["area_sqft"] / 5000.0,
        record["features"]["roof_pitch"] / 10.0
    ]], dtype=torch.float32)


def build_adjacency(num_existing, connect_to=None):
    """
    Dynamically builds adjacency matrix connecting the new entry to existing nodes.
    """
    N = num_existing + 1
    adj = torch.eye(N)

    if connect_to is None:
        connect_to = list(range(num_existing))

    for i in connect_to:
        adj[num_existing, i] = 1
        adj[i, num_existing] = 1

    return adj


def predict_with_new_entry(record):
    """
    Loads existing records, encodes the new entry, builds the combined graph,
    and generates predictions using the trained GAT model.
    Returns predictions scaled back to the original value range.
    """
    x_existing, _ = load_existing_data()
    new_node = encode_new_node(record)
    x_combined = torch.cat([x_existing, new_node], dim=0)
    adj_combined = build_adjacency(len(x_existing))

    model = GATPredictor(x_combined.size(1), HIDDEN_DIM, OUTPUT_DIM)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        preds = model(x_combined, adj_combined)
        result = preds[-1]  # Only the new node prediction

    # Reverse scaling to original scale for meaningful predictions
    return {
        "estimated_cost": round(result[0].item() * 10000, 2),
        "risk_score": round(result[1].item(), 2)
    }


if __name__ == "__main__":
    # Example JSON input for prediction
    input_json = '''
    {
        "features": {
            "num_workers": 2,
            "material_cost": 218100,
            "area_sqft": 500,
            "roof_pitch": 1.0
        }
    }
    '''
    print("Input:", input_json.strip())
    record = json.loads(input_json)
    prediction = predict_with_new_entry(record)
    print("Prediction:", prediction)
