# Irish Food Market Analysis - AI Agent Instructions

## Project Overview
Modular data science project analysing 2000+ Irish food products from OpenFoodFacts. Features ingredient parsing, NOVA classification, TF-IDF recommender system, and interactive Streamlit dashboard.

## Architecture
- **src/harvester.py**: Fetches OpenFoodFacts data, filters for Ireland, upserts to Supabase
- **src/tokenizer.py**: Regex-based ingredient parser handling nested parentheses `r',\s*(?![^()]*\))'`
- **src/recommender.py**: TF-IDF similarity engine + NOVA/additive analysis for healthier alternatives
- **src/clustering.py**: K-Means segmentation, PCA visualisation
- **src/analyser.py**: Nutrient statistics, frequency counts
- **app.py**: Streamlit dashboard with 4 pages (Trends, Recommender, Clustering, About)
- **Food_market_analysis.ipynb**: EDA and hypothesis testing notebook

## Code Patterns

### Ingredient Tokenization (see src/tokenizer.py)
```python
# Split on commas OUTSIDE parentheses only
INGREDIENT_PATTERN = r',\s*(?![^()]*\))'
tokens = re.split(INGREDIENT_PATTERN, text)
```

### Supabase Queries
```python
# Multi-condition OR filters
.or_("field.ilike.%value1%,field.ilike.%value2%")
# Upsert to prevent duplicates
.upsert(data, on_conflict='code')
```

### Recommender System (src/recommender.py)
- **TF-IDF**: `TfidfVectorizer(max_features=100, ngram_range=(1,2))`
- **Similarity**: Cosine similarity on ingredient vectors
- **Filtering**: Exclude NOVA 4, high additive count, concerning E-numbers
- **Scoring**: `processing_score = (nova * 0.4) + (e_count * 0.3) + (additive_score * 0.3)`

### Streamlit Session State
```python
# Initialize ONCE in sidebar (not main page)
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
```

## Build & Test
```bash
# Setup
pip install -r requirements.txt
cp .env.example .env  # Add SUPABASE_URL, SUPABASE_KEY

# Run dashboard
streamlit run app.py

# Test suite
pytest tests/ -v --cov=src --cov-report=term-missing

# Lint
flake8 src/ tests/ --max-line-length=120
```

## CI/CD (GitHub Actions)
- **Triggers**: Push/PR to main/master
- **Jobs**: Lint (flake8), test (pytest + coverage), security check
- **Python**: 3.10, cached pip dependencies

## Data Conventions
- **Missing nutriments**: Default to `0.0` (float)
- **Ireland filter**: `countries_sold.ilike.%Ireland%` OR `countries_en` contains `en:ie`
- **Non-English removal**: Filter out products with German keywords (`zucker`, `wasser`)
- **API pagination**: 50 items/page, max 5 pages per category
- **Required fields**: Skip products missing `ingredients_text` or `code`

## Visualisation Standards
- **Exports**: PNG with `dpi=300, bbox_inches='tight'`
- **Plotly theme**: `template='plotly_white'`
- **Axis labels**: `rotation=45, ha='right'` for ingredient names
- **Colors**: Use `px.colors.qualitative.Safe` for accessibility
<parameter name="filePath">c:\Users\seanq\Desktop\Food_data\.github\copilot-instructions.md