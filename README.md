# Irish Food Market Analysis Pipeline

A full-stack data engineering pipeline that harvests, cleans, and analyses ingredient data from the OpenFoodFacts API, specifically targeting the Irish market.

![Top Ingredients Graph](irish_food_ingredients.png)
*Fig 1: Analysis of the top 20 ingredients found in 1,000+ Irish food products.*

## 🎯 Project Overview
The goal of this project was to determine the "standard composition" of processed food sold in Ireland. 
Raw data from OpenFoodFacts is often user-generated and messy (inconsistent spellings, multilingual tags, formatting errors). This pipeline solves that by standardizing the data into a clean PostgreSQL database for analysis.

## 🛠 Tech Stack
* **Language:** Python 3.9
* **Database:** Supabase (PostgreSQL)
* **ETL & Cleaning:** Pandas, Regex, Custom Tokenizer
* **Visualization:** Seaborn, Matplotlib

## ⚙️ Architecture

### 1. The Harvester (`harvest_to_db.py`)
* Iterates through 12 major food categories (Biscuits, Pizzas, etc.).
* Uses **Pagination** to crawl deep into the API results.
* **Filtering:** Implements specific metadata filters to isolate products sold in `Ireland` or `en:ie`, overcoming the default French/UK bias of the source dataset.
* **Upsert Logic:** Stores data in Supabase with conflict resolution to prevent duplicates.

### 2. The Cleaner (`nlp_lab.py` logic)
* **Regex Tokenization:** Custom Regex pattern `r',\s*(?![^()]*\))'` used to handle complex ingredient lists where commas appear inside parentheses (e.g., *"Acidity Regulators (Citric Acid, Sodium Citrate)"*).
* **Normalization:** Maps synonyms (e.g., *"Flavorings"* -> *"Flavoring"*) to ensure accurate statistical counts.

### 3. The Analyser (`analyse_irish_ingredients.py`)
* Fetches the raw, messy text data from the DB.
* Tokenizes and counts ingredient frequency.
* Generates distribution visualisations using Seaborn.

## 🚀 Key Insights
* **The "Big Two":** Salt and Sugar are present in nearly **50%** of all analyzed products.
* **Oil Trends:** Rapeseed Oil is significantly more common than Palm Oil in the Irish market, likely due to local availability.
* **Hidden Sugars:** High prevalence of industrial sweeteners like *Dextrose* and *Maltodextrin* in savory products.

## 🔮 Future Improvements
* **Nutri-Score vs. Ingredients:** Correlating the number of ingredients with the Nutri-Score grade.
* **Allergen Detection:** Flagging undeclared allergens based on ingredient text analysis.