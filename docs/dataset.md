# 📊 Dataset Documentation

This document explains the structure and meaning of the `dataset.json` file used to train the GAT-based estimator model.

---

## 🧠 Purpose

The dataset simulates job records for a roofing estimation tool. It includes:

* Project features (input)
* Estimation targets (output)
* Graph structure between jobs (edges)

This graph structure enables contextual learning using Graph Attention Networks (GAT), allowing predictions to benefit from patterns in related jobs.

---

## 📁 Structure

Each line in the `dataset.json` is a single JSON object with the following structure:

```json
{
  "nodes": [
    {
      "id": "job1",
      "features": {
        "num_workers": 3,
        "material_cost": 1200.0,
        "area_sqft": 1000.0,
        "roof_pitch": 1.5
      },
      "target": {
        "estimated_cost": 5000.0,
        "risk_score": 0.2
      }
    },
    ...
  ],
  "edges": [
    {"source": "job1", "target": "job2"},
    {"source": "job2", "target": "job3"},
    {"source": "job3", "target": "job4"},
    {"source": "job4", "target": "job5"},
    {"source": "job5", "target": "job1"}
  ]
}
```

---

## 🧾 Node Features

Each job node includes:

| Feature         | Type    | Description                           |
| --------------- | ------- | ------------------------------------- |
| `num_workers`   | Integer | Number of workers assigned to the job |
| `material_cost` | Float   | Raw material cost in USD              |
| `area_sqft`     | Float   | Square footage of the roofing area    |
| `roof_pitch`    | Float   | Pitch of the roof (higher = steeper)  |

### 🎯 Target Outputs

| Target           | Description                        |
| ---------------- | ---------------------------------- |
| `estimated_cost` | Total project cost in USD          |
| `risk_score`     | Risk level score (0.0 – 1.0 scale) |

> 🚧 **Note**: Cost is normalized during training (divided by 100,000).

---

## 🔗 Edges

Edges define relationships between jobs (nodes), forming the graph used for attention-based learning.

### Why Edges Matter

Edges allow each job node to:

* Share contextual features with its neighbors
* Learn richer representations through attention weights
* Improve prediction based on similar jobs (e.g. in size, materials, complexity)

## 🧩 Edge Mapping Context

The current edge definitions are designed to simulate **relational context between job records**, which helps the GAT model learn patterns that go beyond isolated data points.

### 🔄 Current Graph Topology

```json
[
  {"source": "job1", "target": "job2"},
  {"source": "job2", "target": "job3"},
  {"source": "job3", "target": "job4"},
  {"source": "job4", "target": "job5"},
  {"source": "job5", "target": "job1"}
]
```

This forms a **cyclic undirected graph** where each job is directly connected to two others, enabling bidirectional information flow and attention across the structure.

---

### 🤔 Why These Edges?

These specific edge mappings were chosen to:

* Mimic **similarity in job scale** or **chronological progression** (e.g. jobs completed in sequence or with similar teams).
* Ensure **each node has multiple neighbors**, encouraging diverse attention signals.
* Avoid isolated nodes, which cannot benefit from contextual reasoning.

---

### 💡 Example Interpretations

| Connection  | Possible Context                            |
| ----------- | ------------------------------------------- |
| job1 ↔ job2 | Similar crew size and material cost         |
| job2 ↔ job3 | Related complexity or shared contractor     |
| job3 ↔ job4 | Similar region or construction schedule     |
| job4 ↔ job5 | High material usage and roof complexity     |
| job5 ↔ job1 | Wrap-around to reinforce graph connectivity |

---

### 🔍 Edge Policy (for now)

* Undirected (symmetric): If A ↔ B, then B ↔ A
* Equal weight: All edges are treated with equal importance
* Hardcoded: Manually defined for this prototype training dataset


---

## 🛠️ Future Enhancements

* Use real-world Neo4j graph exports
* Add edge types or weights (e.g., similarity score)
* Auto-generate edges via feature-based similarity (cosine or kNN)
