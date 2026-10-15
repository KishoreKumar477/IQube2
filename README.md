# Spotify Tracks Clustering & Analysis (`IQube@.ipynb`)

An end-to-end unsupervised machine learning pipeline for exploratory data analysis, feature preprocessing, clustering, and genre alignment using the **Spotify Tracks Dataset**. The notebook evaluates whether acoustic descriptors alone can partition songs into coherent groups using **K-Means** and **DBSCAN**, and cross-references the clusters against labeled music genres.

---

## Table of Contents
1. [Overview](#overview)
2. [Dataset & Features](#dataset--features)
3. [Pipeline Steps](#pipeline-steps)
4. [Clustering & Quantitative Evaluation](#clustering--quantitative-evaluation)
5. [Genre Alignment & Key Findings](#genre-alignment--key-findings)
6. [Tech Stack & Dependencies](#tech-stack--dependencies)
7. [How to Run](#how-to-run)

---

## Overview

Streaming catalogs use continuous acoustic attributes (tempo, energy, valence, loudness) to characterize audio. This project explores:
- Optimal cluster discovery using inertia (Elbow Method).
- Centroid-based partitioning (**K-Means**) vs. density-based clustering (**DBSCAN**).
- Evaluation via **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Score**.
- 2D dimensionality reduction using **PCA**.
- Cross-tabulation heatmap mapping discovered clusters back to actual track genres.

---

## Dataset & Features

* **Dataset Source:** `maharshipandya/-spotify-tracks-dataset` via KaggleHub
* **Total Records:** 114,000 entries (113,999 non-null after dropping incomplete rows)
* **Target Acoustic Dimensions (15 numeric features):**
  * `popularity`, `duration_ms`, `explicit`
  * `danceability`, `energy`, `key`, `loudness`, `mode`
  * `speechiness`, `acousticness`, `instrumentalness`
  * `liveness`, `valence`, `tempo`, `time_signature`
* **Excluded Metadata Columns:**
  * `Unnamed: 0`, `track_id`, `artists`, `album_name`, `track_name`, `track_genre`

---

## Pipeline Steps

1. **Data Ingestion & Cleaning:**
   * Automated loading via `kagglehub.dataset_load`.
   * Removal of metadata/identifier columns and missing values.
2. **Feature Normalization:**
   * Standardized all 15 numerical features using `StandardScaler` to prevent large-scale dimensions (like `duration_ms` and `tempo`) from dominating distance computations.
3. **Hyperparameter Selection (Elbow Method):**
   * Computed inertia across $k \in [2, 10]$ on a random subset ($N = 10,000$, `random_state=42`), selecting $k=4$ as the balance point.
4. **Clustering Execution:**
   * **K-Means:** $k=4$, `n_init=10`, `random_state=42`.
   * **DBSCAN:** `eps=3.0`, `min_samples=10` to test for density connectivity and isolate outliers.
5. **Dimensionality Reduction:**
   * Projected scaled feature vectors onto 2 principal components (**PCA 1** and **PCA 2**) for side-by-side scatter plot comparisons.
6. **Cross-Tabulation & Genre Mapping:**
   * Re-aligned the sampled cluster assignments with `track_genre`.
   * Filtered for the top 20 most frequent genres and visualized the counts via a Seaborn heatmap.

---

## Clustering & Quantitative Evaluation

All metrics were evaluated on the representative 10,000-track sample:

| Metric | K-Means ($k=4$) | DBSCAN (`eps=3.0`, non-noise) | Criterion |
| :--- | :--- | :--- | :--- |
| **Silhouette Score** | `0.1059` | `0.1879` | Higher is better ($-1$ to $+1$) |
| **Davies-Bouldin Index** | `2.2950` | `1.2290` | Lower is better |
| **Calinski-Harabasz Score** | `1147.50` | `371.96` | Higher is better |

---

## Genre Alignment & Key Findings

* **Continuous Distribution:** The low silhouette score ($~0.11$) demonstrates that audio features exist along a continuous spectrum rather than in distinct, hyper-separated geometric clusters.
* **Density Characteristics:** DBSCAN assigns the vast majority of tracks to a single central density manifold (Cluster 0), categorizing peripheral/extreme tracks as noise (`-1`).
* **Genre Clustering Behavior:**
  * **Cluster 1:** Heavy bias toward high-energy, louder, electronic/heavy styles (`black-metal`, `techno`, `progressive-house`).
  * **Cluster 2:** Strongly aligned with acoustic, complex, or instrumental styles (`jazz`, `indian`, `idm`).
  * **Cluster 0:** Dominated by rhythm-focused and mid-tempo genres (`party`, `blues`, `bluegrass`, `dub`).
  * **Cluster 3:** Isolates high-tempo/percussive tracks with unique acoustic signatures (e.g., `hardcore`).

---

## Tech Stack & Dependencies

```txt
kagglehub[pandas-datasets]
numpy
pandas
matplotlib
seaborn
scikit-learn
