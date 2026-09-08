# Prediction of Concrete Compressive Strength Using Machine Learning

**Course Project — ME644** Machine Learning for Engineers
 Department of Mechanical Engineering, IIT Kanpur

## Overview

Concrete compressive strength is traditionally measured by casting samples and physically breaking them in a lab after 7–28 days of curing — a process that is slow, expensive, and hard to scale across many mix designs. This project builds a machine learning pipeline to predict concrete compressive strength directly from mix composition and curing age, removing the need for a physical lab test every time.

## Dataset

- **Source:** [UCI Machine Learning Repository — Concrete Compressive Strength Dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
- **Original donor:** Prof. I-Cheng Yeh, Chung-Hua University, Taiwan (1998)
- **Size:** 1030 instances, 9 attributes (8 inputs + 1 output)
- **Inputs (all in kg per m³ of mixture, except Age):** Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age (days)
- **Output:** Concrete compressive strength (MPa)

After removing 25 duplicate rows, the cleaned dataset used for analysis contains **1005 unique samples**.

## Methodology

1. **Data Cleaning** — checked for missing values, duplicates, and out-of-range values.
2. **Exploratory Data Analysis** — distributions, boxplots, and correlation analysis of all features against strength.
3. **Feature Scaling** — standardized all 8 input variables (mean 0, std 1) using `StandardScaler`.
4. **PCA** — reduced dimensionality of the (correlated) input variables while retaining ~90% of variance.
5. **K-Means Clustering** — grouped concrete mixes into natural clusters based on PCA-reduced features, profiled by average ingredients and average strength.
6. **Train/Test Split** — 80/20 split, with scaling fit only on the training set to avoid data leakage.
7. **Modeling** — trained and compared three models:
   - Multiple Linear Regression (MLR)
   - K-Nearest Neighbors (KNN)
   - Artificial Neural Network (ANN, `MLPRegressor`)
8. **Evaluation** — compared all models on the held-out test set using R², MAE, and RMSE.
9. **Feature Importance** — MLR coefficients plus permutation importance for KNN and ANN, to identify which ingredients most influence strength.

## Key Findings

**Correlation with strength (EDA):**

| Feature | Correlation |
|---|---|
| Cement | +0.49 |
| Superplasticizer | +0.34 |
| Age | +0.34 |
| Slag | +0.10 |
| Fly Ash | −0.08 |
| Coarse Aggregate | −0.14 |
| Fine Aggregate | −0.19 |
| Water | −0.27 |

**Model performance (test set):**

| Model | R² | MAE (MPa) | RMSE (MPa) |
|---|---|---|---|
| Multiple Linear Regression | 0.580 | 8.896 | 11.192 |
| K-Nearest Neighbors (k=2) | 0.602 | 6.233 | 8.747 |
| Artificial Neural Network (hidden layers = 32,16) | 0.833 | 5.355 | 7.065 |

The ANN clearly outperforms both MLR and KNN — consistent with the well-established fact (from Prof. Yeh's original 1998 paper on this exact dataset) that concrete strength is a highly nonlinear function of its ingredients and age. A deeper ANN architecture (64→32→16 hidden units) was also tested and pushed performance further, to R² = 0.865, MAE = 4.454 — suggesting there's still headroom for improvement with more tuning.

**PCA:** 6 principal components were needed to capture ≥90% of total variance (out of 8 original features).

**K-Means:** best silhouette score (0.297) achieved at k=7 clusters. The clusters revealed clear "recipe families" — e.g. the highest-strength cluster (Cluster 1, avg. strength 52.8 MPa) is characterized by moderate-to-high cement (~400 kg/m³), high superplasticizer content, and relatively low water — while the lowest-strength cluster (Cluster 4, avg. strength 26.0 MPa) uses cement alone with essentially no slag, ash, or superplasticizer.

**Feature importance — consistent across all three models:**

| Feature | MLR Coefficient | KNN Permutation Importance | ANN Permutation Importance |
|---|---|---|---|
| Cement | 12.12 | 0.316 | 1.284 |
| Age | 6.89 | 0.438 | 0.781 |
| Slag | 8.44 | 0.173 | 0.557 |
| Water | −2.84 | 0.159 | 0.075 |
| Superplasticizer | 1.96 | 0.170 | 0.063 |

**Cement, Age, and Slag** consistently emerge as the strongest positive drivers of compressive strength across all three modeling approaches, while **Water** is consistently the strongest negative driver — directly reflecting the classic water-cement ratio principle in concrete engineering.

## Repository Contents

- `concrete_strength_project.ipynb` — full analysis notebook (all 11 steps)
- `Concrete_Data.xls` — original dataset
- `figures/` — saved plots (distributions, correlation heatmap, PCA, clustering, model comparisons, feature importance)

## Tools Used

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook (VS Code)

## Acknowledgements

Dataset originally donated by Prof. I-Cheng Yeh. Reuse permitted with retention of copyright notice and citation of:
> I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," *Cement and Concrete Research*, Vol. 28, No. 12, pp. 1797-1808 (1998).
