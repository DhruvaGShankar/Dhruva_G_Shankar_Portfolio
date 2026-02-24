# Titanic Survival Analysis (EDA + Profiling)

This project performs basic exploratory data analysis on the Titanic dataset, handles missing values, engineers a simple cabin-related feature, and generates an automated profiling report using `ydata-profiling`.

## What it does

- Clones a public datasets repository containing the Titanic CSV
- Loads the dataset into a Pandas DataFrame
- Shows basic dataset info and missing-value counts (`head()`, `info()`, `isna().sum()`)
- Cleans / imputes missing values:
  - Fills missing `Age` with the median (robust to outliers)
  - Fills missing `Embarked` with the mode (most frequent value)
  - Creates `Has_Cabin` = `1` if `Cabin` is present else `0`, then drops `Cabin`
- Generates a full HTML profiling report (`Analysis.html`) using `ydata-profiling`

## Dataset source

The notebook clones:

- `http://github.com/HarshvardhanSingh-13/Datasets`

and reads:

- `/content/Datasets/Titanic_Dataset/Titanic-Dataset.csv`

## Requirements

- Python (Colab works out of the box)
- Libraries:
  - `pandas`, `numpy`
  - `matplotlib`, `seaborn`
  - `ydata-profiling`


## How to run (Google Colab)

1. Clone the datasets repo:
   ```bash
   !git clone http://github.com/HarshvardhanSingh-13/Datasets

2.Install y-data profiling:

```bash
    !pip install ydata-profiling -q
```

3. Run the notebook cells to load, clean, and generate the report.

## Outputs

- **Analysis.html**  
  An automated profiling report (distributions, correlations, missingness, warnings, sample rows)

## Preprocessing notes

- `Age` is filled with median \(28.0\) since median is less affected by outliers than mean
- `Embarked` is filled with mode (commonly `S`) to preserve the most likely category
- `Cabin` is high-missingness and text-based, so it’s converted into a simple indicator feature `Has_Cabin`

