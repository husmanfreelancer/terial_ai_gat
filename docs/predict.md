### 🔮 Real-Time Prediction with GAT

This guide explains how to use the trained **Graph Attention Network (GAT)** model to generate real-time predictions on new job entries — such as **estimated cost** and **risk score** — based on existing job data and graph-based relationships.

---

### 🧠 Objective

Given a new job record (not part of training), the model:

- Embeds the input with existing job data
- Adds it to the graph (connected to prior jobs)
- Predicts:
  - `estimated_cost`
  - `risk_score`

---

### 📁 Input Format

You pass in a single JSON object with the same features used during training:

```json
{
  "features": {
    "num_workers": 2,
    "material_cost": 800,
    "area_sqft": 500,
    "roof_pitch": 1.0
  }
}
````

This is parsed and normalized to match the model’s expectations.

---

### ⚙️ How It Works

1. Loads `real-time-predict.jsonl` as existing job graph context
2. Encodes the new record and appends it to the feature matrix
3. Dynamically constructs a new graph with updated adjacency
4. Uses the trained GAT model (`models/gat_model.pt`) to predict

---

### 🚀 Running Prediction

```bash
python predict.py
```

Sample output:

```
Input: {
  "features": {
    "num_workers": 2,
    "material_cost": 800,
    "area_sqft": 500,
    "roof_pitch": 1.0
  }
}
Prediction: {'estimated_cost': 4083.49, 'risk_score': 0.36}
```

---

### 📂 Files

* `predict.py`: Main real-time prediction script
* `real-time-predict.jsonl`: Existing known jobs for context
* `model.py`: Shared architecture
* `config.py`: Hyperparameters + path to model weights

---

### 🧩 Graph Behavior

* The new job is connected to all existing jobs (default)
* You can modify `build_adjacency()` to connect selectively
* Predictions are made only for the new (appended) node

---

### 💡 Suggestions

* Use this model for **quote estimation**, **job risk review**, or **bid scoring**
* Integrate into live systems with streaming inputs
* Cache predictions and refit the model periodically with fresh data

---

This is part of a complete GAT-based prototype for early-stage estimation intelligence in construction tech, inspired by **Terial**’s use case.
