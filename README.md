# 🕵️‍♀️ The Irish Food Detective

[![CI Pipeline](https://github.com/Synapsean/Irish-Food-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Synapsean/Irish-Food-Analysis/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### 🚀 [Launch Live Dashboard](https://irish-food-analysis-ar7awmwzzgxkfdggawrgp7.streamlit.app/)

**A Data Science Portfolio Project analysing hidden patterns in the Irish Food Supply.**

This project harvests, cleans, and analyses **2,000+ Irish food products** from the OpenFoodFacts API to debunk marketing myths using statistical analysis and Machine Learning.

![Dashboard Screenshot](outputs/dashboard_preview.png)

## ✨ Features

- **🔍 Product Recommender**: TF-IDF ingredient similarity engine finds alternatives to any product
- **🧪 UPF Filter**: Toggle between all products vs ultra-processed foods (NOVA 4)
- **📊 Nutrient Analysis**: Interactive charts for salt, sugar, fat across categories
- **🤖 ML Clustering**: K-Means segmentation reveals 5 ingredient-based market clusters
- **📈 Statistical Testing**: Pingouin t-tests, ANOVA for hypothesis validation

## 📊 Key Insights

| Finding | Evidence |
|---------|----------|
| **The Salt Trap** | Soup products appear 1500% saltier than crisps—but this is skewed by stock cubes (concentrates). Real ready-to-eat soup has **less salt** than crisps. |
| **5 Market Segments** | K-Means clustering revealed 5 distinct product clusters based on ingredient composition, not marketing labels |
| **NOVA Validation** | Cluster analysis aligns with NOVA ultra-processing classification (Silhouette Score: 0.42) |

## ⚠️ Technical Considerations

**Note on Current Implementation:**

This project demonstrates full-stack data science and software engineering capabilities. However, there are known technical limitations in the current NLP approach:

**Semantic Similarity Limitation:**
- Current implementation uses **TF-IDF** (bag-of-words) for ingredient similarity
- Does not capture semantic relationships: "sodium chloride" ≠ "salt", "sugar" ≠ "glucose syrup"
- Industry-standard approach would use **Sentence-BERT embeddings** or similar transformer models

**Evaluation Methodology:**
- Current evaluation uses category-based relevance (circular logic with the filtering mechanism)
- Production system would require **expert-curated gold standard** validation set
- Proper metrics would test actual health improvement, not just category matching

**What This Project Demonstrates:**
- ✅ Full-stack capability: API → Database → ML → Deployment
- ✅ Clean code architecture with modular design and testing
- ✅ CI/CD pipeline with automated testing and quality checks
- ✅ Production deployment experience (live Streamlit app)
- ✅ Data engineering: ETL pipelines, database design, API integration

**Planned Improvements:**
1. Replace TF-IDF with Sentence-Transformers for semantic ingredient matching
2. Implement FAISS/ChromaDB vector database for scalable similarity search
3. Create expert-curated validation set (50+ products with pharmacology-based health rankings)
4. Refactor evaluation to use Precision@K, Recall@K, NDCG@K with non-circular ground truth

### Performance Metrics (Post-Refactoring)

Once semantic embeddings are implemented, the following validation metrics will be calculated:

**Silhouette Score Comparison:**
- Measures clustering quality (values from -1 to 1, higher is better)
- Formula: `s(i) = (b(i) - a(i)) / max(a(i), b(i))`
- Current (TF-IDF K-Means): ~0.42 (reported in insights)
- Expected (SBERT K-Means): >0.55 (semantic clustering should outperform bag-of-words)
- **Status**: Requires Sentence-Transformer implementation

**Recommender System Metrics:**
- Precision@3: What fraction of top-3 recommendations are truly healthier?
- Recall@10: What fraction of all healthier alternatives were captured in top-10?
- NDCG@K: Normalized Discounted Cumulative Gain (rewards ranking quality)
- **Baseline**: Expert-curated gold standard (50 products with pharmacology-validated alternatives)
- **Status**: Requires non-circular evaluation framework

**Semantic Similarity Validation:**
- Cosine similarity: "salt" vs "sodium chloride" (should be >0.8 with SBERT, ~0.0 with TF-IDF)
- Demonstrate improved ingredient matching with embeddings
- **Status**: Requires SBERT implementation

This repository showcases software engineering and deployment skills. The ML sophistication is intentionally kept simple for demonstrative purposes and will be enhanced in future iterations.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Data Collection** | OpenFoodFacts API, Requests |
| **Database** | Supabase (PostgreSQL) |
| **ML/Analysis** | Scikit-Learn (TF-IDF, K-Means, PCA), Pingouin |
| **Visualisation** | Plotly Express, Seaborn |
| **Dashboard** | Streamlit |
| **CI/CD** | GitHub Actions (pytest, flake8) |

## 📂 Project Structure
```
├── src/                  # Core modules
│   ├── harvester.py      # OpenFoodFacts API client
│   ├── tokenizer.py      # Ingredient parsing (regex)
│   ├── clustering.py     # ML pipeline
│   └── recommender.py    # TF-IDF similarity recommender
├── tests/                # Unit tests (pytest)
├── notebooks/            # EDA and hypothesis testing
├── app.py                # Streamlit dashboard
└── .github/workflows/    # CI pipeline
```

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/Synapsean/Irish-Food-Analysis.git
cd Irish-Food-Analysis
pip install -r requirements.txt

# Set up environment
cp .env.example .env  # Add your Supabase credentials

# Run dashboard locally
streamlit run app.py
```

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src
```

---
*Created by [Sean Quinlan](https://linkedin.com/in/sean-quinlan-phd/) | PhD Pharmacology | MSc Data Science @ UCD*
