### 📘 Training the GAT Estimator Model

This guide explains how to train a **Graph Attention Network (GAT)** model to estimate project metrics such as **cost** and **risk score** using real-world construction job data.

---

### 🧠 Objective

Build a predictive model that estimates:

- `estimated_cost` (normalized)
- `risk_score` (scale: 0–1)

from structured job/project features and graph-based relationships using PyTorch.

---

### 📁 Dataset Format (`dataset.json`)

This is a single-line JSON file (not multi-line JSONL), structured as:

- `nodes`: list of jobs with features and labels
- `edges`: relationships between jobs (undirected graph)

#### Example Node:

```json
{
  "id": "job5",
  "features": {
    "num_workers": 6,
    "material_cost": 2200.0,
    "area_sqft": 2000.0,
    "roof_pitch": 3.0
  },
  "label": {
    "estimated_cost": 9200.0,
    "risk_score": 0.5
  }
}
````

#### Example Edge:

```json
{ "source": "job5", "target": "job1" }
```

---

### ⚙️ Feature Engineering

Each job node is converted into a 4D feature vector:

1. `num_workers` (scaled /10)
2. `material_cost` (scaled /10,000)
3. `area_sqft` (scaled /5,000)
4. `roof_pitch` (scaled /10)

Labels are also scaled as:

* `estimated_cost`: divided by 10,000
* `risk_score`: left in raw scale (usually 0–1)

Edges define graph structure; adjacency matrix is symmetric (undirected).

---

### 🚀 Training

#### 1. Environment Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Run Training

```bash
python train.py
```

Model checkpoint will be saved at:

```
models/gat_model.pt
```

---

### 🔍 Files Overview

* `train.py`: Loads dataset, trains GAT model
* `model.py`: Contains `GATLayer` and `GATPredictor`
* `config.py`: Learning rate, epoch count, paths, etc.
* `dataset.json`: Contains project nodes and edge links

---

### 🧭 Future Improvements

* Add **dropout** and **multi-head attention**
* Add **real-time evaluation** and **metric tracking**
* Integrate with **Neo4j** for scalable graph inference
* Enable **fine-tuned GNN explanations** for insight transparency

---

Built to power predictive insights for real-world estimation platforms like **Terial**.
