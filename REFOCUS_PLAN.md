# Food_data Project Pivot - Fewer Ingredients Focus

## Problem Statement
**User's Actual Goal:** *"For my own food consumption I want to eat more food that is less processed and contains less additive and preservatives. So ideally the less ingredients the better."*

**Current Recommender Issue:** 
- Focuses on finding "similar" products (same category)
- Evaluation measures category match, not actual health improvement
- Doesn't prioritize fewer ingredients

---

## New Recommendation Strategy

### Priority Ranking (Most Important → Least):
1. **Fewer Total Ingredients** (40% weight)
2. **Lower Processing (NOVA)** (30% weight)
3. **Fewer E-numbers** (20% weight)
4. **Avoid Concerning Additives** (10% weight)

### Example:
**User Currently Eats:** Kellogg's Corn Flakes (12 ingredients, NOVA 4, 5 E-numbers)

**Old Recommender:** Suggests other breakfast cereals (same category)  
**New Recommender:** Suggests porridge oats (1 ingredient, NOVA 1, 0 E-numbers)

---

## Implementation Plan

### Step 1: Add Ingredient Count Parsing
```python
# src/tokenizer.py - ADD new function

def count_ingredients(ingredients_text):
    """
    Count number of discrete ingredients.
    
    Args:
        ingredients_text (str): Raw ingredient text from OpenFoodFacts
        
    Returns:
        int: Number of ingredients, or None if invalid
        
    Examples:
        >>> count_ingredients("Water, Sugar, Salt")
        3
        >>> count_ingredients("Oats (100%)")
        1
        >>> count_ingredients("Flour (Wheat, Niacin), Sugar, Palm Oil")
        3
    """
    if not ingredients_text or pd.isna(ingredients_text):
        return None
        
    # Use existing tokenizer (handles nested parentheses)
    tokens = tokenize_ingredients(ingredients_text)
    
    # Filter out noise (%, water percentages, etc.)
    real_ingredients = [t for t in tokens if len(t.strip()) > 2]
    
    return len(real_ingredients)


# Add to harvester.py data collection
def clean_product_data(product):
    """Add ingredient_count field during data harvesting."""
    ingredients_text = product.get('ingredients_text', '')
    
    return {
        'code': product['code'],
        'product_name': product['product_name'],
        'ingredients_text': ingredients_text,
        'ingredient_count': count_ingredients(ingredients_text),  # NEW
        'nova_group': product.get('nova_group', 0),
        # ... other fields
    }
```

---

### Step 2: Refactor Recommender Scoring
```python
# src/recommender.py - REPLACE calculate_processing_score()

def calculate_healthiness_score(product):
    """
    Score how 'healthy' a product is based on processing indicators.
    Lower score = better (fewer ingredients, less processed).
    
    Returns:
        float: 0-100 (0=whole food, 100=ultra-processed)
    """
    # Extract features
    ingredient_count = product.get('ingredient_count', 20)  # Default high
    nova = product.get('nova_group', 4)
    e_numbers = count_e_numbers(product.get('ingredients_text', ''))
    concerning = count_concerning_additives(product.get('ingredients_text', ''))
    
    # Normalize to 0-100 scale
    # Ingredient penalty: 0 ingredients=0, 20+ ingredients=100
    ingredient_score = min(ingredient_count / 20 * 100, 100)
    
    # NOVA penalty: NOVA 1=0, NOVA 4=100
    nova_score = (nova - 1) / 3 * 100
    
    # E-number penalty: 0 E-numbers=0, 10+ E-numbers=100
    e_score = min(e_numbers / 10 * 100, 100)
    
    # Concerning additive penalty: 0=0, 5+=100
    additive_score = min(concerning / 5 * 100, 100)
    
    # Weighted combination (prioritize ingredient count)
    final_score = (
        ingredient_score * 0.40 +  # 40% weight on ingredient count
        nova_score * 0.30 +         # 30% weight on processing level
        e_score * 0.20 +            # 20% weight on E-numbers
        additive_score * 0.10       # 10% weight on concerning additives
    )
    
    return round(final_score, 2)


def find_better_alternatives(product_code, df, top_n=10):
    """
    Find healthier alternatives to a given product.
    
    Strategy:
    1. Same category OR similar use case
    2. Lower healthiness_score (fewer ingredients, less processed)
    3. Still available in Ireland
    
    Args:
        product_code (str): Barcode of current product
        df (pd.DataFrame): Full product database
        top_n (int): Number of alternatives to return
        
    Returns:
        pd.DataFrame: Recommended alternatives with improvement scores
    """
    # Get target product
    target = df[df['code'] == product_code].iloc[0]
    target_score = calculate_healthiness_score(target)
    target_category = target.get('categories_en', '')
    
    # Filter candidates
    # - Same broad category (first 2 levels)
    # - Healthier score (lower)
    # - Available in Ireland
    category_root = get_category_root(target_category)
    
    candidates = df[
        (df['categories_en'].str.contains(category_root, case=False, na=False)) &
        (df['code'] != product_code) &
        (df['countries_sold'].str.contains('Ireland', case=False, na=False))
    ].copy()
    
    # Calculate scores
    candidates['healthiness_score'] = candidates.apply(calculate_healthiness_score, axis=1)
    
    # Filter: Must be healthier (lower score)
    candidates = candidates[candidates['healthiness_score'] < target_score]
    
    # Calculate improvement
    candidates['improvement'] = target_score - candidates['healthiness_score']
    candidates['improvement_pct'] = (candidates['improvement'] / target_score * 100).round(1)
    
    # Sort by improvement (best alternatives first)
    candidates = candidates.sort_values('improvement', ascending=False)
    
    # Add explanations
    candidates['why_better'] = candidates.apply(
        lambda x: generate_explanation(target, x), axis=1
    )
    
    return candidates.head(top_n)[
        ['product_name', 'ingredient_count', 'nova_group', 'healthiness_score', 
         'improvement', 'improvement_pct', 'why_better']
    ]


def generate_explanation(target, alternative):
    """Human-readable explanation of why alternative is better."""
    reasons = []
    
    ing_target = target.get('ingredient_count', 20)
    ing_alt = alternative.get('ingredient_count', 20)
    if ing_alt < ing_target:
        reasons.append(f"{ing_target - ing_alt} fewer ingredients")
    
    if alternative['nova_group'] < target['nova_group']:
        reasons.append(f"Less processed (NOVA {alternative['nova_group']} vs {target['nova_group']})")
    
    e_target = count_e_numbers(target.get('ingredients_text', ''))
    e_alt = count_e_numbers(alternative.get('ingredients_text', ''))
    if e_alt < e_target:
        reasons.append(f"{e_target - e_alt} fewer E-numbers")
    
    if not reasons:
        return "Similar processing, different brand"
    
    return " | ".join(reasons)


def get_category_root(category_str):
    """Extract broad category (e.g., 'Dairy' from 'Dairy > Cheese > Cheddar')."""
    if not category_str:
        return ""
    parts = category_str.split('>')
    return parts[0].strip() if parts else ""
```

---

### Step 3: Update Streamlit Dashboard
```python
# app.py - UPDATE Recommender page

def page_recommender():
    st.title("🔄 Find Healthier Alternatives")
    
    st.markdown("""
    **Goal:** Find products with fewer ingredients and less processing.
    
    **How it works:**
    1. Select a product you currently buy
    2. See alternatives with:
       - Fewer total ingredients
       - Lower processing (NOVA score)
       - Fewer additives and E-numbers
    """)
    
    # Load data
    df = load_data()
    df['healthiness_score'] = df.apply(calculate_healthiness_score, axis=1)
    
    # Product search
    product_name = st.selectbox(
        "Search for a product:",
        options=df['product_name'].dropna().unique()
    )
    
    if st.button("Find Better Alternatives"):
        product_code = df[df['product_name'] == product_name]['code'].iloc[0]
        alternatives = find_better_alternatives(product_code, df, top_n=10)
        
        if len(alternatives) == 0:
            st.warning("No healthier alternatives found in database. This might be a good choice already!")
        else:
            st.success(f"Found {len(alternatives)} healthier alternatives:")
            
            # Display comparison table
            st.dataframe(alternatives, use_container_width=True)
            
            # Visualize improvement
            fig = px.bar(
                alternatives.head(5),
                x='product_name',
                y='improvement_pct',
                title='Top 5 Alternatives by Health Improvement',
                labels={'improvement_pct': 'Health Score Improvement (%)'},
                color='improvement_pct',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig)
```

---

## Step 4: New Evaluation Metrics

### Replace Category-Based Metrics with Health Improvement Metrics

```python
# src/recommender_evaluation.py - REPLACE evaluation logic

def evaluate_recommender(df, test_products, k=10):
    """
    Evaluate recommender on real health improvement.
    
    Metrics:
    1. Average Ingredient Reduction
    2. Average NOVA Improvement
    3. Average E-number Reduction
    4. Healthiness Score Improvement
    
    Args:
        df: Full product database
        test_products: List of product codes to test
        k: Number of recommendations to evaluate
        
    Returns:
        dict: Evaluation metrics
    """
    results = {
        'ingredient_reduction': [],
        'nova_improvement': [],
        'e_number_reduction': [],
        'score_improvement': []
    }
    
    for product_code in test_products:
        target = df[df['code'] == product_code].iloc[0]
        alternatives = find_better_alternatives(product_code, df, top_n=k)
        
        if len(alternatives) > 0:
            # Average improvements across top-k recommendations
            ing_reduction = (
                target.get('ingredient_count', 20) - 
                alternatives['ingredient_count'].mean()
            )
            results['ingredient_reduction'].append(ing_reduction)
            
            nova_improvement = (
                target.get('nova_group', 4) - 
                alternatives['nova_group'].mean()
            )
            results['nova_improvement'].append(nova_improvement)
            
            e_target = count_e_numbers(target.get('ingredients_text', ''))
            e_alt_mean = alternatives.apply(
                lambda x: count_e_numbers(x['ingredients_text']), axis=1
            ).mean()
            results['e_number_reduction'].append(e_target - e_alt_mean)
            
            score_improvement = alternatives['improvement'].mean()
            results['score_improvement'].append(score_improvement)
    
    # Aggregate metrics
    return {
        'avg_ingredient_reduction': np.mean(results['ingredient_reduction']),
        'avg_nova_improvement': np.mean(results['nova_improvement']),
        'avg_e_number_reduction': np.mean(results['e_number_reduction']),
        'avg_score_improvement': np.mean(results['score_improvement']),
        'n_tested': len(test_products)
    }


# Example Usage
test_set = [
    '5410188031850',  # Kellogg's Corn Flakes
    '5449000000996',  # Coca-Cola
    '3017620422003',  # Nutella
    # ... add more common products
]

metrics = evaluate_recommender(df, test_set, k=10)
print(f"Average ingredient reduction: {metrics['avg_ingredient_reduction']:.1f}")
print(f"Average NOVA improvement: {metrics['avg_nova_improvement']:.2f}")
print(f"Average healthiness improvement: {metrics['avg_score_improvement']:.1f}%")
```

---

## Implementation Timeline

### Week 1 (Days 1-3):
- [x] Document problem and strategy (this file)
- [ ] Add `count_ingredients()` to tokenizer.py
- [ ] Update harvester.py to calculate ingredient_count
- [ ] Regenerate database with ingredient counts
- [ ] Test ingredient_count accuracy on 50 products

### Week 1 (Days 4-7):
- [ ] Refactor recommender.py with healthiness_score
- [ ] Implement find_better_alternatives()
- [ ] Update Streamlit dashboard
- [ ] Manual testing: Does it find genuinely better alternatives?

### Week 2 (Days 8-14):
- [ ] Build new evaluation metrics
- [ ] Compare old vs. new recommender on test set
- [ ] Document improvements in README
- [ ] Deploy updated dashboard

---

## Expected Improvements

### Before (Category-Based):
**User input:** Kellogg's Corn Flakes  
**Recommendations:** Other branded cereals (Nesquick, Frosties, etc.)  
**Problem:** Still ultra-processed, many ingredients

### After (Ingredient-Count-Based):
**User input:** Kellogg's Corn Flakes  
**Recommendations:**
1. Plain porridge oats (1 ingredient)
2. Shredded wheat (2 ingredients)
3. Weetabix (4 ingredients)

**Benefit:** Actually helps you eat less processed food

---

## Success Criteria

1. **Functionality:** Recommender prioritizes products with fewer ingredients
2. **Accuracy:** Ingredient count parser works on 95%+ of products
3. **Utility:** User can scan barcode, get better alternatives in 10 seconds
4. **Validation:** Average recommended product has 5+ fewer ingredients than input

---

## Next Steps

1. Implement count_ingredients() function
2. Update database with ingredient counts
3. Refactor recommender scoring
4. Test on your actual shopping list
5. Iterate based on real-world usage

---

**Key Insight:** This is a *personal tool* for healthier eating, not an academic project. Success = "Does it help me make better food choices?" not "Does it predict categories well?"
