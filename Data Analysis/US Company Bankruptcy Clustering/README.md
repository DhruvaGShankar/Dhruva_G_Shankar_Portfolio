# US Company Bankruptcy Clustering Analysis 🏢📉

This project applies unsupervised machine learning techniques to segment American companies based on their financial metrics. By utilizing clustering algorithms, the analysis identifies distinct financial profiles—ranging from highly distressed to financially robust—which can serve as a foundational step for bankruptcy prediction and risk assessment.

## 📋 Overview

The script automates the retrieval of financial data directly from Kaggle, performs extensive data cleaning and feature dimensionality reduction, and applies K-Means clustering. Finally, it uses Principal Component Analysis (PCA) to visualize the distinct segments of companies and cross-references these clusters with their actual bankruptcy status.

## 🧰 Tech Stack & Libraries

* **Python 3**
* **Kagglehub**: For seamless programmatic downloading of the dataset.
* **Pandas & NumPy**: For data manipulation, aggregation, and mathematical operations.
* **Scikit-Learn (`sklearn`)**: For machine learning pipeline steps, including `StandardScaler`, `VarianceThreshold`, `KMeans`, and `PCA`.
* **Matplotlib & Seaborn**: For visualizing the elbow method and PCA scatter plots.

## 🗄️ Dataset

The data is sourced from Kaggle using the `kagglehub` library:
* **Dataset Name**: `american-companies-bankruptcy-prediction-dataset`
* **Author**: `utkarshx27`
* **File**: `american_bankruptcy.csv`

## 🧹 Data Cleaning & Dimensionality Reduction

Given the complexity and potential noise in financial datasets, a strict preprocessing pipeline was implemented:

* **Missing Value Handling**: Columns with more than **40%** missing data were dropped. Remaining missing values were imputed using the feature's median to avoid outlier distortion.
* **Standardization**: Features were scaled using `StandardScaler` to ensure metrics with different units (e.g., ratios vs. raw dollar amounts) contribute equally to the distance calculations in clustering.
* **Low-Variance Filtering**: A `VarianceThreshold` (0.01) was applied to remove constant or near-constant features that provide little predictive power.
* **Multicollinearity Removal**: Highly correlated features (Pearson correlation > **0.85**) were identified and dropped to simplify the model and prevent redundant data from skewing the clusters.

---

## 🤖 Modeling & Key Insights

The core of the analysis relies on K-Means clustering to group companies. 

### 1. Determining the Number of Clusters (Elbow Method)
* **Approach**: The model calculated the within-cluster Sum of Squared Errors (SSE) / Inertia for $k$ values from 1 to 10. * **Insight**: The inertia decreases sharply from $k=1$ to $k \approx 3-4$, after which the rate of improvement slows significantly. This indicates that most of the meaningful structure in the data is captured by **3 clusters**, beyond which additional clusters yield diminishing returns.

### 2. Cluster Profiling
Companies were segmented into three distinct groups based on their financial health:
* **Cluster 0**: High-Risk Distressed Firms
* **Cluster 1**: Moderately Stable Firms
* **Cluster 2**: Strong Financial Health

### 3. Visualizing with PCA
* **Approach**: Principal Component Analysis (PCA) was used to reduce the high-dimensional feature space into 2 principal components for 2D visualization. * **Insight**: The clustering reveals three well-defined segments of companies with progressively increasing PC1 values. The leftmost cluster is compact and homogeneous, indicating distressed firms share highly similar (and restricted) financial structures. Conversely, the rightmost cluster is much more dispersed, suggesting greater variability and flexibility among larger or financially healthier profiles.

## 🚀 How to Run

1. Clone or download the repository containing the script.
2. Ensure you have the required libraries installed:
   ```bash
   pip install kagglehub pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the script in your preferred Python environment (e.g., Jupyter, Colab, or terminal). It will automatically fetch the dataset from Kaggle, execute the preprocessing pipeline, and display the Elbow and PCA plots.
