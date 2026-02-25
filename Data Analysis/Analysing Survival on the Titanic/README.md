# Analyzing Survival on the Titanic 🚢🧊

This project focuses on cleaning, preprocessing, and performing Exploratory Data Analysis (EDA) on the classic Titanic dataset. The goal is to prepare the data for potential predictive modeling by handling missing values and generating a comprehensive, interactive data profile report.

## 📋 Overview

Using Python and the `ydata-profiling` library, this script automates the exploratory data analysis process. It addresses missing data using statistical methods and transforms sparse columns into usable features, ultimately outputting an interactive HTML report detailing the dataset's characteristics, correlations, and distributions.

## 🧰 Tech Stack & Libraries

* **Python 3**
* **Pandas & NumPy**: For data manipulation, feature engineering, and statistical imputation.
* **Matplotlib & Seaborn**: For foundational data visualization support.
* **YData Profiling (`ydata-profiling`)**: For generating automated, detailed, and interactive EDA reports.

## 🗄️ Dataset

The dataset used is `Titanic-Dataset.csv`, sourced from the following repository:
* [HarshvardhanSingh-13/Datasets](http://github.com/HarshvardhanSingh-13/Datasets)

## 🧹 Data Cleaning & Preprocessing

A significant portion of this project revolves around handling missing data strategically to avoid skewing the dataset:

* **Age Imputation**: Missing values in the `Age` column were filled using the **median** age (28.0). The median was selected over the mean because it is more robust and unaffected by outliers in the age distribution.
* **Embarked Imputation**: The `Embarked` column (port of embarkation) had a few missing values. These were filled using the **mode** ('S'). Because the vast majority of passengers embarked at 'S', adding a few more rows to this category had a negligible impact on the overall distribution.
* **Cabin Feature Engineering**: The `Cabin` column contained a massive amount of missing data, making imputation impractical. Instead of dropping the information entirely, it was converted into a binary feature called `Has_Cabin` (1 if a cabin was listed, 0 if missing). The original `Cabin` column was then dropped.

---

## 💡 Exploratory Data Analysis (EDA)

Instead of manually plotting every variable, this project leverages **YData Profiling** to generate an automated, exhaustive analysis. 

The script generates an interactive report (`Analysis.html`) that provides:
* **Overview Statistics**: Total variables, missing cells, and duplicate rows.
* **Variable Properties**: Distributions, distinct counts, and descriptive statistics for features like Fare, Age, and Survival.
* **Interactions & Correlations**: Heatmaps and scatter plots showing how variables relate to one another (e.g., Class vs. Survival, Gender vs. Survival).
* **Missing Values Analysis**: Visual matrices detailing data completeness.

## 🚀 How to Run

1. Clone the repository containing the script.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn ydata-profiling
   ```
3. Run the script. It will automatically download the dataset via `git clone`, process the data, and generate `Analysis.html` in your working directory. Open this HTML file in any web browser to view the interactive report.
