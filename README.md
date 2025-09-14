# K-Means Clustering Algorithm

## Overview

This repository provides a comprehensive explanation of the K-Means clustering algorithm, including theoretical foundations, mathematical formulations, implementation examples, and methods for selecting the optimal number of clusters.

## Table of Contents

- [Theoretical Explanation](#theoretical-explanation)
- [Mathematical Formulation](#mathematical-formulation)
- [Selecting the Number of Clusters (K)](#selecting-the-number-of-clusters-k)
- [Implementation](#implementation)
- [Usage Examples](#usage-examples)
- [Requirements](#requirements)
- [Installation](#installation)

---

## Theoretical Explanation

K-Means is an **unsupervised clustering** algorithm that partitions data into **K groups** where points within a group are more similar to each other than to those in other groups.

**Goal:** Minimize the **intra-cluster variance** by grouping similar data points together.

### Algorithm Steps (Lloyd's Algorithm)

1. **Initialization**: Choose K initial centroids (randomly or using K-Means++)
2. **Assignment Step**: Assign each data point to the nearest centroid (using Euclidean distance)
3. **Update Step**: Recalculate centroids as the mean of all points in each cluster
4. **Repeat**: Steps 2 and 3 until centroids stabilize or maximum iterations are reached

### Advantages

- Simple and computationally efficient
- Works well with spherical clusters of similar sizes
- Scales well to large datasets
- Guaranteed convergence

### Disadvantages

- Requires manual selection of **K**
- Sensitive to initialization and outliers
- Assumes clusters are spherical and of similar size
- Performs poorly with non-spherical or varying density clusters

---

## Mathematical Formulation

Given a dataset **X = {x₁, x₂, ..., xₙ}** where **xᵢ ∈ ℝᵈ**, we aim to partition **X** into **K** clusters **C = {C₁, C₂, ..., Cₖ}** to minimize the objective function:

```
J = Σₖ₌₁ᴷ Σₓᵢ∈Cₖ ||xᵢ - μₖ||²
```

Where:
- **μₖ = (1/|Cₖ|) Σₓᵢ∈Cₖ xᵢ** (centroid of cluster Cₖ)
- **||·||** is the Euclidean norm

### Assignment Step

Assign **xᵢ** to cluster **Cₖ** if:
```
||xᵢ - μₖ||² ≤ ||xᵢ - μⱼ||² for all j
```

### Update Step

Update centroid:
```
μₖⁿᵉʷ = (1/|Cₖ|) Σₓᵢ∈Cₖ xᵢ
```

**Time Complexity**: O(n · K · d · iterations)

---

## Selecting the Number of Clusters (K)

### A) Elbow Method

Plot **inertia** (sum of squared distances to nearest centroid) vs. K and choose the "elbow" point where inertia decreases slowly.

```
Inertia = Σₖ₌₁ᴷ Σₓᵢ∈Cₖ ||xᵢ - μₖ||²
```

### B) Silhouette Score

Measures how similar a point is to its own cluster compared to other clusters. For each point **i**:

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
```

Where:
- **a(i)**: Average distance between **i** and all points in its cluster
- **b(i)**: Minimum average distance between **i** and points in other clusters
- **Range**: [-1, 1], higher values indicate better clustering

### C) Gap Statistic

Compares the total intra-cluster variation for different values of K with their expected values under null reference distribution of the data.

---

## Implementation

### Basic K-Means with Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
y_kmeans = kmeans.fit_predict(X)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X', label='Centroids', edgecolors='black')
plt.title('K-Means Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Inertia: {kmeans.inertia_:.2f}")
```

### Elbow Method Implementation

```python
from sklearn.metrics import silhouette_score

def find_optimal_k(X, max_k=10):
    """Find optimal number of clusters using Elbow Method and Silhouette Score"""
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_k + 1)  # Start from 2 for silhouette score
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        y_pred = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, y_pred))
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Elbow Method
    ax1.plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal K')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette Score
    ax2.plot(K_range, silhouette_scores, marker='s', color='orange', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal K based on silhouette score
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"Optimal K based on Silhouette Score: {optimal_k}")
    
    return K_range, inertias, silhouette_scores

# Example usage
K_range, inertias, silhouette_scores = find_optimal_k(X, max_k=10)
```

---

## Usage Examples

### Example 1: Basic Clustering

```python
# Load your data
# X = your_data_here

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

print(f"Number of clusters: {kmeans.n_clusters}")
print(f"Inertia: {kmeans.inertia_:.2f}")
```

### Example 2: Real-world Dataset

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load iris dataset
iris = load_iris()
X = iris.data

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Evaluate clustering
from sklearn.metrics import adjusted_rand_score
ari_score = adjusted_rand_score(iris.target, labels)
print(f"Adjusted Rand Index: {ari_score:.3f}")
```

---

## Requirements

```
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd k-means-clustering
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the examples:
   ```bash
   python kmeans_example.py
   ```

---

## Best Practices

- **Data Preprocessing**: Always standardize/normalize your features before applying K-Means
- **Initialization**: Use multiple random initializations (`n_init` parameter) to avoid poor local minima
- **Evaluation**: Use multiple metrics (inertia, silhouette score, ARI) to evaluate clustering quality
- **Visualization**: Always visualize your results when possible (use PCA for high-dimensional data)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
