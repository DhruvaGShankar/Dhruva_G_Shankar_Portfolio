# House Price Prediction (Advanced Regression) 🏠💰

This project builds a robust predictive model to estimate the sale prices of residential homes. By applying comprehensive exploratory data analysis (EDA), strategic feature engineering, and an XGBoost regression model, this script processes raw housing data into highly accurate price predictions suitable for Kaggle competitions.

## 📋 Overview

The analysis automates the end-to-end machine learning pipeline: from authenticating and downloading the dataset via the Kaggle API, to cleaning heavily missing data, engineering new impactful features, and finally training an optimized gradient boosting model to predict the final continuous target variable (`SalePrice`).

## 🧰 Tech Stack & Libraries

* **Python 3**
* **XGBoost**: For the core gradient-boosted regression model.
* **Scikit-Learn (`sklearn`)**: For data splitting, scaling (`StandardScaler`), and evaluation metrics (RMSE, MAE, $R^2$).
* **Pandas & NumPy**: For extensive data manipulation, one-hot encoding, and mathematical transformations (e.g., log transformations).
* **Matplotlib & Seaborn**: For visualizing distributions and feature correlations.

## 🗄️ Dataset

The dataset is sourced from the popular Kaggle competition:
* **Dataset Name**: `house-prices-advanced-regression-techniques`

## 🧹 Data Preprocessing & EDA

Real-world housing data is often messy and skewed. The following steps were taken to clean and prepare the data for modeling:

* **Target Variable Transformation**: The target variable (`SalePrice`) exhibited significant right-skewness. To normalize the distribution and improve model performance, a logarithmic transformation (`np.log1p`) was applied. * **Correlation Analysis**: A correlation matrix was generated to identify the top 10 numeric features most heavily correlated with the sale price (e.g., Overall Quality, Living Area). * **Strategic Imputation**: 
  * Missing numerical values related to physical footprint (e.g., `BsmtFinSF1`, `GarageArea`) were imputed with `0`, assuming the property lacks that feature.
  * `LotFrontage` was intelligently imputed using the median value of the specific `Neighborhood` the house is located in.
  * Missing categorical values were filled with `'None'` (if the feature likely didn't exist, like a Pool) or the mode (most frequent value) for standard features.

## 🏗️ Feature Engineering

To give the model stronger predictive signals, several new features were engineered from the existing data:
* **`TotalSF`**: Combined basement, first-floor, and second-floor square footage into a single metric for the total living area.
* **`TotalBath`**: Aggregated full baths, half baths (weighted at 0.5), and basement baths into a single bathroom count.
* **`Age`**: Calculated the age of the house at the time of sale (`YrSold` - `YearBuilt`).
* **Categorical Encoding**: Converted all categorical text columns into numerical format using one-hot encoding (`pd.get_dummies`).

---

## 🤖 Modeling & Evaluation

* **Model**: **XGBoost Regressor** (`xgb.XGBRegressor`). 
* **Hyperparameters**: The model was tuned with a slow learning rate (`0.05`), `1000` estimators, and a max depth of `3` to prevent overfitting while capturing complex non-linear relationships.
* **Evaluation**: The model was evaluated on a 20% validation holdout set using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared ($R^2$). 
* **Final Submission**: The predictions on the unseen test set were transformed back from log-scale to standard dollar amounts using `np.expm1` and formatted into `submission.csv` for Kaggle scoring.

## 🚀 How to Run

1. Clone or download the repository.
2. Ensure you have your `kaggle.json` API token ready to upload when prompted by the script.
3. Install the required libraries:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
   ```
4. Run the script in a Jupyter environment (like Google Colab). It will automatically configure the Kaggle API, download the data, train the model, and output the final `submission.csv`.
