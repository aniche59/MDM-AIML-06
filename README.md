# MDM Practical Assignment 06: Unsupervised Clustering Algorithms & Performance Benchmark

**Student Name:** Aditya Iche  
**PRN:** 202401100136  
**Branch:** Computer Science (Software Engineering)  
**Division:** B | **Batch:** B2 | **Roll No:** 48  
**Faculty In-Charge:** Mr. Khushal Khairnar  

---

## 📌 Problem Overview & Objectives
This repository implements, benchmarks, and comparatively analyzes four foundational unsupervised machine learning clustering paradigms using the **Mall Customer Segmentation Benchmark Dataset** (200 customer profiles, features: Age, Annual Income in $k, Spending Score 1–100):

1. **K-Means Clustering**: Centroid-based partitioning algorithm that minimizes the Within-Cluster Sum of Squares (WCSS / Inertia) using Lloyd's heuristic and $k$-means++ centroid initialization.
2. **Agglomerative Hierarchical Clustering**: Bottom-up connectivity-based clustering constructing a taxonomic dendrogram using Euclidean distance and Ward's minimum variance criterion to minimize within-cluster variance at each merge step.
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**: Non-parametric density-based algorithm that discovers clusters of arbitrary shapes and isolates low-density noise/outliers without requiring an a priori specification of cluster count ($k$).
4. **Gaussian Mixture Model (GMM)**: Generative probabilistic soft-clustering model that represents the overall data distribution as a convex linear superposition of Gaussian probability density functions, optimized iteratively via the Expectation-Maximization (EM) algorithm.

---

## 📐 Mathematical Formulations

### 1. K-Means Objective Function (Within-Cluster Sum of Squares / Inertia)
$$J_{K\text{-Means}} = \sum_{k=1}^{K} \sum_{x_i \in C_k} \| x_i - \mu_k \|^2$$
$$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$
$$\text{Update Rule: } C_k^{(t)} = \left\{ x_i : \|x_i - \mu_k^{(t)}\|^2 \le \|x_i - \mu_j^{(t)}\|^2, \forall j \in \{1, \dots, K\} \right\}$$

### 2. Ward's Hierarchical Minimum Variance Linkage
$$\Delta \text{ESS}_{AB} = \frac{|A| \cdot |B|}{|A| + |B|} \| \mu_A - \mu_B \|^2$$
$$d(A, B) = \sum_{x \in A \cup B} \| x - \mu_{A \cup B} \|^2 - \left( \sum_{x \in A} \| x - \mu_A \|^2 + \sum_{y \in B} \| y - \mu_B \|^2 \right)$$

### 3. DBSCAN Density Reachability & Core Point Criteria
$$N_\epsilon(p) = \{ q \in D \mid \text{dist}(p, q) \le \epsilon \}$$
$$\text{Core Point Condition: } |N_\epsilon(p)| \ge \text{MinPts}$$
$$\text{Border Point Condition: } |N_\epsilon(p)| < \text{MinPts} \quad \text{and} \quad \exists q \in N_\epsilon(p) \text{ such that } q \text{ is a Core Point}$$
$$\text{Noise / Outlier: } |N_\epsilon(p)| < \text{MinPts} \quad \text{and} \quad \forall q \in N_\epsilon(p), q \text{ is not a Core Point}$$

### 4. Gaussian Mixture Model (EM Probabilistic Likelihood)
$$P(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k), \quad \sum_{k=1}^{K} \pi_k = 1$$
$$\text{E-step: } \gamma_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$
$$\text{M-step: } \mu_k = \frac{\sum_{i=1}^{N} \gamma_{ik} x_i}{\sum_{i=1}^{N} \gamma_{ik}}, \quad \Sigma_k = \frac{\sum_{i=1}^{N} \gamma_{ik} (x_i - \mu_k)(x_i - \mu_k)^T}{\sum_{i=1}^{N} \gamma_{ik}}, \quad \pi_k = \frac{1}{N} \sum_{i=1}^{N} \gamma_{ik}$$

### 5. Cluster Validity Metrics

#### a) Silhouette Coefficient ($s(i)$)
$$a(i) = \frac{1}{|C_A| - 1} \sum_{j \in C_A, j \ne i} d(i, j), \quad b(i) = \min_{C_B \ne C_A} \frac{1}{|C_B|} \sum_{j \in C_B} d(i, j)$$
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$
$$\text{Silhouette Score} = \frac{1}{N} \sum_{i=1}^{N} s(i)$$

#### b) Davies-Bouldin Index (DBI)
$$R_{ij} = \frac{s_i + s_j}{d(\mu_i, \mu_j)}, \quad \text{where } s_i = \frac{1}{|C_i|} \sum_{x \in C_i} \|x - \mu_i\|$$
$$DB = \frac{1}{K} \sum_{i=1}^{K} \max_{j \ne i} R_{ij} \quad (\text{Lower is Better})$$

#### c) Calinski-Harabasz Index (CHI / Variance Ratio Criterion)
$$CH = \frac{\text{Trace}(B_K) / (K - 1)}{\text{Trace}(W_K) / (N - K)} \quad (\text{Higher is Better})$$
$$B_K = \sum_{k=1}^{K} n_k (\mu_k - \mu)(\mu_k - \mu)^T, \quad W_K = \sum_{k=1}^{K} \sum_{x \in C_k} (x - \mu_k)(x - \mu_k)^T$$

---

## 📊 Quantitative Results & Comparative Analysis

| Algorithm | Clusters ($K$) | Noise Points | Silhouette Score | Davies-Bouldin Index | Calinski-Harabasz Index | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **K-Means ($k=5$)** | **5** | **0** | **0.5547** | **0.5722** | **248.65** | 82.56 ms |
| **Agglomerative (Ward, $k=5$)** | **5** | **0** | 0.5538 | 0.5779 | 244.41 | 10.86 ms |
| **DBSCAN ($\epsilon=0.38, \text{min}=5$)** | 4 | 17 | 0.4884 | 0.5811 | 153.56 | **8.04 ms** |
| **Gaussian Mixture (GMM, $k=5$)** | **5** | **0** | 0.5537 | 0.5760 | 244.94 | 33.69 ms |

---

## 👥 Discovered Consumer Personas ($K=5$)

From the Mall Customer benchmark (Annual Income vs Spending Score), the models reliably identified 5 commercially distinct consumer personas:

1. **Target VIP Consumers (High Income, High Spend)**: Premium retail targets with high discretionary purchasing power and high brand engagement.
2. **Careful Frugal Consumers (High Income, Low Spend)**: Affluent individuals who exercise financial prudence and price consciousness; require value-driven incentives.
3. **Careless Impulse Buyers (Low Income, High Spend)**: Younger demographics or impulse shoppers with high spending despite lower income; responsive to micro-financing and seasonal trends.
4. **Sensible Budget Shoppers (Low Income, Low Spend)**: Frugal consumers focused on essential commodities, discounts, and clearance items.
5. **Mainstream Consumers (Average Income, Average Spend)**: The largest balanced middle cohort providing reliable, consistent store traffic.

---

## 🔍 Algorithmic Suitability for Diverse Clustering Tasks

1. **Globular, Convex & Equi-Dispersed Partitions (e.g. Market Segmentation, Vector Quantization)**:
   - **Winner**: **K-Means ($k=5$)**.
   - *Rationale*: Achieved the highest Silhouette Score (0.5547), lowest Davies-Bouldin Index (0.5722), and highest Calinski-Harabasz ratio (248.65). It establishes clean, spherical Voronoi partitions ideal for compact, well-separated consumer clusters.

2. **Hierarchical & Nested Taxonomies (e.g. Phylogenetics, Document Ontologies)**:
   - **Winner**: **Agglomerative Hierarchical Clustering (Ward's Linkage)**.
   - *Rationale*: Generates a deterministic, interpretable dendrogram tree structure without requiring random initialization. Matches K-Means quality (Silhouette: 0.5538) while allowing dynamic post-hoc threshold selection at any hierarchical level.

3. **Noisy Environments with Outliers & Arbitrary Geometric Shapes (e.g. Spatial Geofencing, Anomaly Detection)**:
   - **Winner**: **DBSCAN**.
   - *Rationale*: Automatically isolated 17 extreme outlier noise points without forcing them into artificial cluster centroids. Its $O(N \log N)$ spatial indexing makes it the fastest solver (8.04 ms) and eliminates the vulnerability to cluster center contamination.

4. **Overlapping & Probabilistic Soft Assignments (e.g. Multi-Interest User Profiling, Topic Modeling)**:
   - **Winner**: **Gaussian Mixture Models (GMM)**.
   - *Rationale*: Accommodates elliptical covariance matrices ($\Sigma_k$) with arbitrary orientation and provides soft probabilistic assignment weights ($\gamma_{ik}$), capturing uncertainty along boundary regions between consumer segments.

---

## 📈 Visualizations & Artifacts Generated

- **`elbow_and_silhouette_analysis.png`**: Dual-panel validation displaying the WCSS Elbow point and Silhouette score curve across $k \in [2, 10]$ confirming $k=5$ as the optimal partition.
- **`clusters_2d_visualization.png`**: $2 \times 2$ high-resolution grid depicting spatial partitions, centroids, component means, and noise points across all 4 algorithms.
- **`clustering_metrics_comparison.png`**: Multi-panel comparative bar chart contrasting Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
- **`hierarchical_dendrogram.png`**: Complete agglomerative hierarchical dendrogram using Ward's minimum variance linkage with the cluster cut-off threshold at $k=5$.
- **`dbscan_kdistance_noise.png`**: 5-NN distance graph for $\epsilon$ knee thresholding alongside the spatial classification of Core, Border, and Noise points.
- **`terminal_output.png`**: High-resolution IDE terminal console render showing execution benchmarks, cluster counts, and validation metrics.

---

## 🛠️ Installation & Execution

```bash
# Clone repository
git clone https://github.com/aniche59/MDM-AIML-06.git
cd MDM-AIML-06

# Install dependencies
pip install -r requirements.txt

# Run unsupervised clustering benchmark
python clustering_algorithms_benchmark.py
```

---
