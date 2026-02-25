# Netflix Content Analysis 🎬📊

This project performs an Exploratory Data Analysis (EDA) on the Netflix Titles dataset to uncover trends, content distribution, and catalog characteristics over time. 

## 📋 Overview

The analysis uses Python to clean, process, and visualize Netflix's content data. By exploring relationships between release dates, addition dates, content ratings, and descriptions, this project highlights how Netflix's content strategy has evolved.

## 🧰 Tech Stack & Libraries

* **Python 3**
* **Pandas & NumPy**: For data manipulation and numerical operations.
* **Matplotlib & Seaborn**: For creating static data visualizations.
* **WordCloud**: For text-based visual analysis of content descriptions.

## 🗄️ Dataset

The dataset used is `netflix_titles.csv`, sourced from the following repository:
* [HarshvardhanSingh-13/Datasets](https://github.com/HarshvardhanSingh-13/Datasets)

## 🧹 Data Cleaning & Preprocessing

Before diving into the analysis, the dataset required several preprocessing steps to handle missing values and format dates:
* **Imputation**: Missing values in the `director` and `cast` columns were filled with `'Unknown'`. Missing `country` values were imputed using the mode (most frequent country).
* **Dropping Nulls**: Rows with missing `date_added` or `rating` were dropped, as they represent a tiny fraction of the dataset and are difficult to accurately impute.
* **Feature Engineering**: The `date_added` column was converted to a datetime format to extract two new features: `year_added` and `month_added`. Additionally, a `content_age` column was created by subtracting the `release_year` from the `year_added`.

---

## 💡 Key Insights & Exploratory Data Analysis (EDA)

The analysis answers five primary questions about the Netflix catalog:

### 1. How has the distribution of content ratings changed over time?
* **Insight:** After 2014, the share of TV-MA rated content grew rapidly, peaking around 2019. It remains the most prevalent rating for both movies and TV shows. TV-14 content also increased, but at a more gradual pace. This trend highlights a distinct shift toward more mature, adult-oriented programming on the platform.

![Distribution of Content Over Time] (Data Analysis/Netflix Content Analysis/img/download (1).png)

### 2. Is there a relationship between content age and its type (Movie vs. TV Show)?
* **Insight:** Netflix prioritizes recent releases, especially for TV shows, where most titles are added within roughly 20 years of their initial release. Movies exhibit a much broader age range (up to ~60 years old), indicating that Netflix still maintains a robust catalog of classic films alongside its newer content.

### 3. Are there trends in content production (Release Year vs. Year Added)?
* **Insight:** Earlier additions to the platform mostly consisted of older, existing catalog titles. However, recent additions heavily skew toward newly released content, showcasing Netflix's strategic pivot toward fresh content and original productions.

### 4. What are the most common themes in content descriptions?
* **Insight:** A WordCloud generated from title descriptions reveals a strong focus on relationships ("family", "father", "mother", "daughter", "wife"), personal journeys ("life", "love", "new", "world"), and conflicted themes ("murder", "secret", "help", "save"). This suggests the catalog relies heavily on character-driven narratives, emotional drama, and high-stakes themes.

### 5. Who are the top directors on Netflix?
* **Insight:** The distribution of titles across directors is highly fragmented. Even the most prolific directors contribute only a modest number of titles overall. This indicates that Netflix builds its catalog using a broad, diverse range of creators rather than relying heavily on a select few individuals.

---

## 🚀 How to Run

1. Clone the repository containing the script.
2. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy matplotlib seaborn wordcloud
   ```
3. Run the script in a Jupyter environment or as a standard Python file. The script will automatically fetch the dataset via `git clone`.
