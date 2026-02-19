# Maintenance Guide - Irish Food Detective

This guide covers routine maintenance tasks to keep the project current and functioning.

## Table of Contents
1. [Monthly Data Refresh](#monthly-data-refresh)
2. [Data Quality Monitoring](#data-quality-monitoring)
3. [Dependency Updates](#dependency-updates)
4. [Dashboard Health Check](#dashboard-health-check)
5. [GitHub Actions Status](#github-actions-status)

---

## Monthly Data Refresh

**Frequency:** Monthly (or quarterly if low activity)  
**Time Required:** 10-15 minutes

### Why?
OpenFoodFacts is continuously updated by contributors. Products are added, updated, or removed. Regular refreshes keep our analysis current.

### Steps

1. **Update all categories:**
```bash
python scripts/update_database.py
```

2. **Update specific categories only:**
```bash
python scripts/update_database.py --categories "Biscuits,Cereal,Yogurt"
```

3. **Full refresh (deletes and rebuilds):**
```bash
python scripts/update_database.py --full-refresh
```

4. **Check update log:**
```bash
cat outputs/logs/database_update_*.txt | tail -n 50
```

### Expected Results
- 2,000-2,500 total products
- 50-300 products per category
- Some categories (e.g., Biscuits, Cereals) will have more data
- Log file saved to `outputs/logs/`

### Troubleshooting
- **"No products found"**: OpenFoodFacts API may be down. Wait 1 hour and retry.
- **"Too many requests"**: Rate limit hit. Wait 5 minutes between retries.
- **Supabase connection error**: Check `.env` file has correct `SUPABASE_URL` and `SUPABASE_KEY`

---

## Data Quality Monitoring

**Frequency:** After each data refresh  
**Time Required:** 5 minutes

### Why?
Ensures incoming data meets quality standards and flags issues early.

### Steps

1. **Generate quality report:**
```bash
python scripts/data_quality_report.py
```

2. **Review the report sections:**
   - **Missing Data**: Should be < 30% for critical fields (ingredients, NOVA)
   - **Data Quality Issues**: Check for duplicates, negative values, outliers
   - **Nutrient Statistics**: Verify values are realistic
   - **Recommendations**: Action items for data improvement

3. **Check for red flags:**
   - ❌ > 50% missing ingredients → Need better API filters
   - ❌ Duplicate product codes → Harvester logic issue
   - ❌ Impossible values (e.g., salt > 100g/100g) → Data validation needed

### Expected Quality Metrics
- Ingredients available: > 70%
- Nutrient data completeness: > 50%
- NOVA classification: > 60%
- No negative nutrient values
- No impossible values (> 100g/100g)

---

## Dependency Updates

**Frequency:** Quarterly (or when security alerts appear)  
**Time Required:** 20-30 minutes

### Why?
Keeps dependencies secure and compatible with latest Python versions.

### Steps

1. **Check for outdated packages:**
```bash
pip list --outdated
```

2. **Update non-breaking packages:**
```bash
pip install --upgrade pandas numpy scikit-learn plotly streamlit
pip freeze > requirements.txt
```

3. **Run tests to ensure nothing broke:**
```bash
pytest tests/ -v
```

4. **Check for security vulnerabilities:**
```bash
pip install safety
safety check -r requirements.txt
```

5. **Update GitHub Actions if Python version changes:**
```yaml
# .github/workflows/ci.yml
python-version: '3.11'  # Update if needed
```

### Critical Dependencies to Watch
- `streamlit`: Can have breaking changes between major versions
- `scikit-learn`: Models may need retraining after updates
- `supabase`: API changes can break database queries
- `plotly`: Chart rendering might change

---

## Dashboard Health Check

**Frequency:** Weekly (or after major changes)  
**Time Required:** 5 minutes

### Why?
Ensures the live Streamlit app is functioning correctly.

### Steps

1. **Visit live dashboard:**
   - URL: https://irish-food-analysis-ar7awmwzzgxkfdggawrgp7.streamlit.app/

2. **Test critical features:**
   - [ ] Page loads without errors
   - [ ] "Trends" tab shows charts
   - [ ] "Recommender" returns results
   - [ ] "Clustering" visualizes correctly
   - [ ] "About" page renders

3. **Check error logs in Streamlit Cloud:**
   - Go to Streamlit Cloud dashboard
   - Check "Logs" for recent errors
   - Look for:
     - Database connection errors
     - API timeouts
     - Memory issues

4. **Verify data freshness:**
   - Check when data was last updated (shown in dashboard)
   - Should match latest database refresh

### Common Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| "Supabase connection error" | Expired credentials | Update secrets in Streamlit Cloud settings |
| "Out of memory" | Too much data loaded at once | Implement pagination or data caching |
| Charts not rendering | Plotly version mismatch | Pin plotly version in requirements.txt |
| Slow load times | Large dataset | Add `@st.cache_data` decorators |

---

## GitHub Actions Status

**Frequency:** After each commit  
**Time Required:** 2 minutes

### Why?
CI/CD pipeline catches bugs before they reach production.

### Steps

1. **Check workflow status:**
   - Go to: https://github.com/[your-username]/Irish-Food-Analysis/actions
   - Latest commit should show green ✓

2. **Review failed workflows:**
   - Click on failed workflow
   - Read error message
   - Common failures:
     - Linting errors (flake8)
     - Test failures (pytest)
     - Security vulnerabilities (safety check)

3. **Fix and re-push:**
```bash
# Fix linting
flake8 src/ tests/ --max-line-length=120

# Fix tests
pytest tests/ -v

# Commit and push
git add .
git commit -m "Fix: CI pipeline errors"
git push
```

### CI Pipeline Components
- **Linting (flake8)**: Code style compliance
- **Testing (pytest)**: Unit tests must pass
- **Coverage**: Aim for > 70% code coverage
- **Security**: Checks for known vulnerabilities

---

## Emergency Procedures

### Dashboard is Down
1. Check Streamlit Cloud status page
2. Verify `.env` secrets are still valid
3. Check GitHub repo for recent breaking changes
4. Rollback to last working commit if needed

### Database Connection Lost
1. Verify Supabase is online (status.supabase.com)
2. Check API keys haven't expired
3. Regenerate keys if necessary (update `.env` and Streamlit secrets)

### Data Corruption Detected
1. Stop all data harvesting immediately
2. Restore from last known good backup
3. Review harvester logic for bugs
4. Run quality report to verify restoration

---

## Automation Opportunities

### Future Improvements
1. **Scheduled data refreshes**: GitHub Actions cron job to run `update_database.py` monthly
2. **Automated quality reports**: Email digest of quality metrics
3. **Dashboard uptime monitoring**: Pingdom or StatusCake alerts
4. **Dependency update bot**: Dependabot for automated PR creation

### Example GitHub Action for Monthly Refresh
```yaml
# .github/workflows/monthly_refresh.yml
name: Monthly Database Refresh

on:
  schedule:
    - cron: '0 3 1 * *'  # 3 AM on 1st of each month
  workflow_dispatch:

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: python scripts/update_database.py
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
      - run: python scripts/data_quality_report.py
      - name: Commit updated logs
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add outputs/logs/
          git commit -m "Auto: Monthly data refresh" || echo "No changes"
          git push
```

---

## Questions or Issues?

Contact: sean.quinlan91@gmail.com  
GitHub Issues: https://github.com/Synapsean/Irish-Food-Analysis/issues
