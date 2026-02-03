# Irish Food Market Analysis - AI Agent Instructions

## Project Overview
Data pipeline analyzing Irish food market from OpenFoodFacts API. Harvests data to Supabase, cleans ingredients with regex tokenization, performs statistical analysis and ML clustering.

## Architecture
- **Harvester** (`harvest_to_db.py`): Fetches OpenFoodFacts data for 12 categories, filters for Ireland, upserts to Supabase
- **Cleaner** (`nlp_lab.py`): Tokenizes ingredients using regex `r',\s*(?![^()]*\))'` to handle parentheses
- **Analyzer** (`analyse_irish_ingredients.py`): Counts frequencies, normalizes synonyms, visualizes with Seaborn
- **ML Analysis** (`Food_market_analysis.ipynb`): TF-IDF vectorization, KMeans clustering, hypothesis testing

## Key Patterns
- **Supabase Queries**: Use `.or_("field.ilike.%value1%,field.ilike.%value2%")` for multi-condition filters
- **Ingredient Tokenization**: Split on commas but skip those inside parentheses using negative lookahead
- **Normalization**: Apply synonym mappings (e.g., `{'flavourings': 'flavouring'}`) before counting
- **Data Cleaning**: Convert nutriments to float, default 0.0 for missing values
- **Filtering Non-English**: Remove products containing German keywords like 'zucker', 'wasser'
- **Upsert Logic**: Use Supabase upsert to prevent duplicates on 'code' field

## Dependencies & Environment
- Python 3.9+ in virtual environment (`venv/`)
- Load credentials from `.env`: `SUPABASE_URL`, `SUPABASE_KEY`
- Key packages: pandas, supabase-py, scikit-learn, seaborn, matplotlib, pingouin, yellowbrick

## Workflows
- **Data Harvesting**: Run `python harvest_to_db.py` (respects API rate limits with sleeps)
- **Analysis**: Execute notebook cells sequentially for ML workflows
- **Visualization**: Use Seaborn barplots with `rotation=45, ha='right'` for ingredient labels
- **Stats Testing**: Pingouin t-tests with `correction='auto'` for unequal variances

## Conventions
- Store plots as PNG with `dpi=300, bbox_inches='tight'`
- Filter Ireland data: `countries_sold.ilike.%Ireland%` or `en:ie`
- Handle API pagination: 50 items/page, up to 5 pages per category
- Skip products without `ingredients_text` or `code`</content>
<parameter name="filePath">c:\Users\seanq\Desktop\Food_data\.github\copilot-instructions.md