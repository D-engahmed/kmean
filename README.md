# K-Means Clustering Algorithm

## Overview
This Jupyter notebook provides a comprehensive explanation of the K-Means clustering algorithm, including theoretical foundations, mathematical formulations, implementation examples, and methods for selecting the optimal number of clusters.

## Table of Contents
1. [Theoretical Explanation](#theoretical-explanation)
2. [Mathematical Formulation](#mathematical-formulation)
3. [Selecting the Number of Clusters (K)](#selecting-the-number-of-clusters-k)
4. [Code Implementation](#code-implementation)
5. [Elbow Method Implementation](#elbow-method-implementation)

---

## Theoretical Explanation
K-Means is an **unsupervised clustering** algorithm that partitions data into **K groups** where points within a group are more similar to each other than to those in other groups.

**Goal:** Minimize the **intra-cluster variance**.

### Algorithm Steps (Lloyd's Algorithm):
1. **Initialization**: Choose K initial centroids (randomly or using K-Means++).
2. **Assignment Step**: Assign each data point to the nearest centroid (using Euclidean distance).
3. **Update Step**: Recalculate centroids as the mean of all points in each cluster.
4. **Repeat**: Steps 2 and 3 until centroids stabilize or max iterations are reached.

**Advantages**:
- Simple and fast.
- Works well with spherical clusters of similar sizes.

**Disadvantages**:
- Requires manual selection of **K**.
- Sensitive to outliers.
- Performs poorly with non-spherical clusters.

---

## Mathematical Formulation
Given a dataset \( X = \{x_1, x_2, \dots, x_n\}, \quad x_i \in \mathbb{R}^d \), we aim to partition \( X \) into \( K \) clusters \( C = \{C_1, C_2, \dots, C_K\} \) to minimize the objective function:

\[
J = \sum_{k=1}^K \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
\]

Where:
- \( \mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i \) (centroid of cluster \( C_k \))
- \( \| \cdot \| \) is the Euclidean norm

**Assignment Step**:
\[
\text{Assign } x_i \text{ to } C_k \quad \text{if} \quad \|x_i - \mu_k\|^2 \leq \|x_i - \mu_j\|^2 \quad \forall j
\]

**Update Step**:
\[
\mu_k^{\text{new}} = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i
\]

**Time Complexity**: \( O(n \cdot K \cdot d \cdot \text{iterations}) \)

---

## Selecting the Number of Clusters (K)
### A) Elbow Method
- Plot **inertia** (sum of squared distances to nearest centroid) vs. K.
- Choose the "elbow" point where inertia decreases slowly.

\[
\text{Inertia} = \sum_{k=1}^K \sum_{x_i \in C_k} \| x_i - \mu_k \|^2
\]

### B) Silhouette Score
Measures how similar a point is to its own cluster vs. other clusters. For each point \( i \):

\[
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\]

Where:
- \( a(i) \): Average distance between \( i \) and all points in its cluster.
- \( b(i) \): Minimum average distance between \( i \) and points in other clusters.
- Range: \([-1, 1]\), higher is better.

---

## Code Implementation
### Basic K-Means + Visualization
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply K-Means
kmeans = KMeans(n_clusters=4, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='red', marker='X', label='Centroids')
plt.legend()
plt.show()
```

**Visualization Meaning**:
- Each color represents a cluster.
- Red `X` marks represent centroids.

---

## Elbow Method Implementation
```python
inertia = []
K_range = range(1, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.plot(K_range, inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()
```

**Interpretation**: Choose K where the inertia curve forms an "elbow" (point of diminishing returns).

---

## Conclusion
This notebook covers the essentials of K-Means clustering, from theory to practical implementation. Use the provided code examples to experiment with synthetic data and apply the Elbow Method to determine the optimal number of clusters for your datasets.
