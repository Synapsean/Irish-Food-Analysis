# 🕵️‍♀️ The Irish Food Detective

### 🚀 [Launch Live Dashboard](https://irish-food-analysis-ar7awmwzzgxkfdggawrgp7.streamlit.app/)

**A Data Science Portfolio Project analysing hidden patterns in the Irish Food Supply.**

This project scrapes, cleans, and analyses 2,000+ Irish food products to debunk marketing myths using statistical analysis and Machine Learning.

## 📊 Key Insights
* **The Salt War:** Discovered that "Soup" products often contain **1500% more salt** than expected due to hidden "concentrate" outliers (stock cubes).
* **Ingredient Clustering:** Used **TF-IDF** and **K-Means Clustering** to segment the market based on ingredient composition rather than marketing labels.

## 🛠️ Tech Stack
* **Data Collection:** Custom Python Scraper (OpenFoodFacts API)
* **Database:** Supabase (PostgreSQL)
* **Analysis:** Pandas, Scikit-Learn (PCA, K-Means), Pingouin
* **Visualisation:** Plotly Express, Streamlit
* **Deployment:** Streamlit Cloud

## 📂 Project Structure
* `src/`: Core logic for data harvesting and cleaning.
* `notebooks/`: Exploratory Data Analysis (EDA) and hypothesis testing.
* `app.py`: The production dashboard code.

---
*Created by Sean Q. | [View on LinkedIn](#)*
