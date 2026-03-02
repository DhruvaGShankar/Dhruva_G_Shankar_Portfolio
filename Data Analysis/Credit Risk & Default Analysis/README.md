# Credit Risk & Default Analysis 🏦📊

This project performs an in-depth Exploratory Data Analysis (EDA) and risk assessment on Lending Club loan data. The analysis aims to uncover the primary drivers of loan default, identify high-loss loan segments, and determine the safest, most profitable borrower profiles to optimize lending strategies.

## 📋 Overview

Using Python, this script automates data retrieval from Kaggle, rigorously cleans and encodes categorical financial data, and extracts insights. By combining standard statistical visualizations with a Random Forest Classifier to assess feature importance, this project bridges the gap between descriptive analytics and predictive risk modeling.

## 🧰 Tech Stack & Libraries

* **Python 3**
* **Kagglehub**: For programmatic downloading of the dataset directly from Kaggle.
* **Pandas & NumPy**: For extensive data wrangling, feature engineering, and aggregations.
* **Scikit-Learn (`sklearn`)**: Specifically `RandomForestClassifier` for modeling feature importance.
* **Matplotlib & Seaborn**: For generating bar charts, scatter plots, and complex heatmaps (like the Net Profitability Map).

## 🗄️ Dataset

The data is sourced from Kaggle using the `kagglehub` library:
* **Dataset Name**: `lending-club-loan-preprocessed-dataset`
* **Author**: `gabrielsantello`
* **File**: `lending_club_loan_two.csv`

## 🧹 Data Cleaning & Preprocessing

Financial datasets require meticulous formatting before analysis. The script performs several key transformations:

* **Handling Missing Data**: Rows with negligible missing values (e.g., employment title, revolving utilization) were dropped to ensure a perfectly clean dataset without introducing imputation bias.
* **Text to Numeric Parsing**: Extracted integer values from the `term` string (e.g., "36 months" converted to `36`).
* **Ordinal Encoding**: Mapped categorical rankings into logical numeric scales:
  * `emp_length` mapped from categorical strings to continuous numbers (0 to 10+ years).
  * `grade` and `sub_grade` mapped into numeric values to reflect credit quality deterioration logically.
* **Target Variable Creation**: The `loan_status` column was converted into a binary target variable `loan_repaid` (1 for 'Fully Paid', 0 for 'Charged Off').
* **Temporal Features**: Calculated the age of the borrower's credit line (`earliest_cr_line`) by subtracting the year of origination from 2026.
* **One-Hot Encoding**: Used `pd.get_dummies` to transform remaining nominal categories (home ownership, loan purpose, etc.) into binary features for machine learning.

---

## 💡 Key Insights & Exploratory Data Analysis

The project answers five critical business questions for lenders:

### 1. Which borrower characteristics most strongly predict default?
* **Approach:** A `RandomForestClassifier` was trained on the preprocessed data to extract feature importances.
* **Insight:** Default risk is heavily tied to financial strain. The strongest predictors of a charge-off are high Debt-to-Income (DTI) ratios, large revolving balances, and high overall credit utilization.

### 2. Which loan segments generate the highest losses?
* **Insight:** Mid-tier credit grades (specifically **Grades C and D**) generate the highest aggregate losses. This is where high loan volume intersects dangerously with elevated default risk.

### 3. How does default risk change across loan term and Grade?
* **Insight:** Risk compounds based on both quality and time. As expected, default rates rise as credit grades worsen. Furthermore, within *every* single grade category, **60-month loans exhibit significantly higher default rates** than 36-month loans.

### 4. Are high-interest loans compensating for higher risk or just masking it?
* **Approach:** Calculated a "Net Profitability Map" by subtracting the default rate from the average interest rate for specific grade/term combinations.
* **Insight:** High interest rates largely fail to compensate for the extreme default risk of lower-grade loans. **Only top-grade borrowers (Grade A) on shorter terms (36 months) generate solid positive net returns.** Longer, lower-grade loans are functionally unprofitable.

### 5. What borrower profile represents the safest lending opportunity?
* **Insight:** Default risk is heavily correlated with housing status and income. The safest lending opportunity lies with **high-income homeowners holding a mortgage**. Conversely, very low-income renters present the highest statistical probability of default.

## 🚀 How to Run

1. Clone or download the repository containing the script.
2. Ensure you have the required libraries installed:
   ```bash
   pip install kagglehub pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the script. It will automatically download the dataset from Kaggle, execute the data processing pipeline, and plot the analytical charts.
