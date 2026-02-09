"""
Healthier Alternative Recommender Module

Finds less-processed alternatives to ultra-processed foods (UPFs).
Uses NOVA classification, additive counts, and ingredient analysis.
"""

import re
from typing import List, Dict, Optional, Tuple
import pandas as pd

# =============================================================================
# ADDITIVE DETECTION
# =============================================================================

# E-number regex pattern (E100-E1999)
E_NUMBER_PATTERN = re.compile(r'\bE\d{3,4}[a-z]?\b', re.IGNORECASE)

# Known concerning additives (not exhaustive, for demonstration)
CONCERNING_ADDITIVES: Dict[str, str] = {
    # Artificial sweeteners
    'aspartame': 'Artificial sweetener (E951)',
    'acesulfame': 'Artificial sweetener (E950)',
    'sucralose': 'Artificial sweetener (E955)',
    'saccharin': 'Artificial sweetener (E954)',
    
    # Preservatives
    'sodium benzoate': 'Preservative (E211)',
    'potassium sorbate': 'Preservative (E202)',
    'sodium nitrite': 'Preservative linked to health concerns (E250)',
    'sodium nitrate': 'Preservative (E251)',
    
    # Colours
    'tartrazine': 'Artificial colour (E102)',
    'sunset yellow': 'Artificial colour (E110)',
    'carmoisine': 'Artificial colour (E122)',
    'brilliant blue': 'Artificial colour (E133)',
    'caramel colour': 'Colour additive (E150)',
    
    # Flavour enhancers
    'monosodium glutamate': 'Flavour enhancer (E621)',
    'disodium guanylate': 'Flavour enhancer (E627)',
    'disodium inosinate': 'Flavour enhancer (E631)',
    
    # Emulsifiers & thickeners
    'carrageenan': 'Thickener with disputed safety (E407)',
    'xanthan gum': 'Thickener (E415)',
    'polysorbate': 'Emulsifier (E432-E436)',
    
    # Acids
    'phosphoric acid': 'Acidifier, may affect calcium absorption (E338)',
    'citric acid': 'Acidifier (E330)',
    
    # Other
    'modified starch': 'Processed starch',
    'maltodextrin': 'Highly processed carbohydrate',
    'hydrogenated': 'Trans fats indicator',
    'palm oil': 'Environmental/health concerns',
}


def count_e_numbers(ingredients_text: str) -> int:
    """Count E-numbers in ingredient text."""
    if not ingredients_text:
        return 0
    matches = E_NUMBER_PATTERN.findall(ingredients_text)
    return len(matches)


def find_concerning_additives(ingredients_text: str) -> List[Tuple[str, str]]:
    """
    Find known concerning additives in ingredient text.
    
    Returns:
        List of (additive_name, description) tuples
    """
    if not ingredients_text:
        return []
    
    text_lower = ingredients_text.lower()
    found = []
    
    for additive, description in CONCERNING_ADDITIVES.items():
        if additive in text_lower:
            found.append((additive, description))
    
    return found


def calculate_processing_score(row: pd.Series) -> int:
    """
    Calculate a "processing score" (0-100, higher = more processed).
    
    Combines:
    - NOVA group (40% weight)
    - E-number count (30% weight)  
    - Concerning additives (30% weight)
    """
    score = 0
    
    # NOVA component (0-40 points)
    nova = row.get('nova_group')
    if nova:
        try:
            nova_int = int(nova)
            score += (nova_int - 1) * 13  # NOVA 1=0, NOVA 4=39
        except (ValueError, TypeError):
            score += 20  # Unknown = medium
    else:
        score += 20
    
    # E-number component (0-30 points)
    e_count = count_e_numbers(str(row.get('ingredients_text', '')))
    score += min(e_count * 5, 30)  # Cap at 30
    
    # Concerning additives component (0-30 points)
    additives = find_concerning_additives(str(row.get('ingredients_text', '')))
    score += min(len(additives) * 6, 30)  # Cap at 30
    
    return min(score, 100)


def get_nova_label(nova_group: Optional[int]) -> str:
    """Get human-readable NOVA label."""
    labels = {
        1: "Unprocessed/Minimal",
        2: "Processed Culinary",
        3: "Processed",
        4: "Ultra-Processed"
    }
    try:
        return labels.get(int(nova_group), "Unknown")
    except (ValueError, TypeError):
        return "Unknown"


def get_processing_badge(score: int) -> Tuple[str, str]:
    """
    Get emoji badge and label based on processing score.
    
    Returns:
        (emoji, label) tuple
    """
    if score <= 20:
        return "🟢", "Minimally Processed"
    elif score <= 40:
        return "🟡", "Lightly Processed"
    elif score <= 60:
        return "🟠", "Moderately Processed"
    else:
        return "🔴", "Highly Processed"


# =============================================================================
# ALTERNATIVE FINDER
# =============================================================================

def find_healthier_alternatives(
    product: Dict,
    df: pd.DataFrame,
    top_n: int = 3,
    same_category_only: bool = True
) -> List[Dict]:
    """
    Find less-processed alternatives to a given product.
    
    Args:
        product: Dict with product info (must have 'product_name', 'category_searched', etc.)
        df: Full product dataframe
        top_n: Number of alternatives to return
        same_category_only: If True, only search within same category
    
    Returns:
        List of alternative products with comparison data
    """
    product_name = product.get('product_name', '')
    category = product.get('category_searched', '')
    
    # Calculate processing score for input product
    product_series = pd.Series(product)
    input_score = calculate_processing_score(product_series)
    input_nova = product.get('nova_group')
    
    # Filter candidates
    candidates = df.copy()
    
    # Exclude the input product itself
    candidates = candidates[candidates['product_name'] != product_name]
    
    # Filter by category if requested
    if same_category_only and category:
        candidates = candidates[candidates['category_searched'] == category]
    
    # Calculate processing scores for all candidates
    candidates['processing_score'] = candidates.apply(calculate_processing_score, axis=1)
    
    # Only keep products with LOWER processing score
    candidates = candidates[candidates['processing_score'] < input_score]
    
    # If no strictly better alternatives, look for equal or better
    if len(candidates) == 0:
        candidates = df[df['product_name'] != product_name].copy()
        if same_category_only and category:
            candidates = candidates[candidates['category_searched'] == category]
        candidates['processing_score'] = candidates.apply(calculate_processing_score, axis=1)
        candidates = candidates[candidates['processing_score'] <= input_score]
    
    # Sort by processing score (lowest first)
    candidates = candidates.sort_values('processing_score')
    
    # Prepare output
    alternatives = []
    for _, row in candidates.head(top_n).iterrows():
        alt_score = row['processing_score']
        badge, label = get_processing_badge(alt_score)
        
        alternatives.append({
            'product_name': row['product_name'],
            'brand': row.get('brand', 'Unknown'),
            'category': row.get('category_searched', 'Unknown'),
            'nova_group': row.get('nova_group'),
            'nova_label': get_nova_label(row.get('nova_group')),
            'processing_score': alt_score,
            'processing_badge': badge,
            'processing_label': label,
            'score_improvement': input_score - alt_score,
            'e_numbers': count_e_numbers(str(row.get('ingredients_text', ''))),
            'concerning_additives': find_concerning_additives(str(row.get('ingredients_text', ''))),
            'sugar_100g': row.get('sugar_100g', 0),
            'salt_100g': row.get('salt_100g', 0),
        })
    
    return alternatives


def get_product_analysis(product: Dict) -> Dict:
    """
    Get full analysis of a single product.
    
    Returns dict with:
    - processing_score
    - processing_badge/label
    - nova info
    - e_numbers count
    - concerning_additives list
    - improvement suggestions
    """
    product_series = pd.Series(product)
    score = calculate_processing_score(product_series)
    badge, label = get_processing_badge(score)
    
    ingredients = str(product.get('ingredients_text', ''))
    e_numbers = count_e_numbers(ingredients)
    additives = find_concerning_additives(ingredients)
    
    # Generate suggestions based on what's found
    suggestions = []
    if e_numbers > 3:
        suggestions.append(f"Contains {e_numbers} E-numbers - look for products with fewer additives")
    if any('sweetener' in desc.lower() for _, desc in additives):
        suggestions.append("Contains artificial sweeteners - try naturally sweetened alternatives")
    if any('colour' in desc.lower() for _, desc in additives):
        suggestions.append("Contains artificial colours - look for naturally coloured options")
    if 'palm oil' in ingredients.lower():
        suggestions.append("Contains palm oil - consider products with other vegetable oils")
    
    return {
        'product_name': product.get('product_name'),
        'brand': product.get('brand'),
        'category': product.get('category_searched'),
        'nova_group': product.get('nova_group'),
        'nova_label': get_nova_label(product.get('nova_group')),
        'processing_score': score,
        'processing_badge': badge,
        'processing_label': label,
        'e_numbers': e_numbers,
        'concerning_additives': additives,
        'sugar_100g': product.get('sugar_100g', 0),
        'salt_100g': product.get('salt_100g', 0),
        'suggestions': suggestions,
    }


# =============================================================================
# CATEGORY INSIGHTS
# =============================================================================

def get_category_stats(df: pd.DataFrame, category: str) -> Dict:
    """Get processing statistics for a category."""
    cat_df = df[df['category_searched'] == category].copy()
    
    if len(cat_df) == 0:
        return {}
    
    cat_df['processing_score'] = cat_df.apply(calculate_processing_score, axis=1)
    
    return {
        'category': category,
        'product_count': len(cat_df),
        'avg_processing_score': round(cat_df['processing_score'].mean(), 1),
        'min_processing_score': cat_df['processing_score'].min(),
        'max_processing_score': cat_df['processing_score'].max(),
        'avg_sugar': round(cat_df['sugar_100g'].mean(), 2),
        'avg_salt': round(cat_df['salt_100g'].mean(), 2),
        'cleanest_product': cat_df.loc[cat_df['processing_score'].idxmin(), 'product_name'],
    }


# =============================================================================
# CLI DEMO
# =============================================================================

if __name__ == "__main__":
    import os
    from supabase import create_client, Client
    from dotenv import load_dotenv
    
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    supabase: Client = create_client(url, key)
    
    print("Loading data...")
    response = supabase.table("food_items") \
        .select("*") \
        .or_("countries_sold.ilike.%Ireland%,countries_sold.ilike.%en:ie%") \
        .neq("ingredients_text", "") \
        .limit(2000) \
        .execute()
    
    df = pd.DataFrame(response.data)
    print(f"Loaded {len(df)} products\n")
    
    # Demo: Analyze a random ultra-processed product
    upf_products = df[df['nova_group'] == 4].head(5)
    
    for _, product in upf_products.iterrows():
        print("=" * 60)
        analysis = get_product_analysis(product.to_dict())
        print(f"Product: {analysis['product_name']}")
        print(f"Processing: {analysis['processing_badge']} {analysis['processing_label']} (Score: {analysis['processing_score']})")
        print(f"NOVA: {analysis['nova_group']} ({analysis['nova_label']})")
        print(f"E-numbers: {analysis['e_numbers']}")
        
        if analysis['concerning_additives']:
            print("Concerning additives:")
            for name, desc in analysis['concerning_additives'][:3]:
                print(f"  - {name}: {desc}")
        
        print("\n🔄 Finding healthier alternatives...")
        alternatives = find_healthier_alternatives(product.to_dict(), df, top_n=2)
        
        for i, alt in enumerate(alternatives, 1):
            print(f"\n  Alternative {i}: {alt['product_name']}")
            print(f"  {alt['processing_badge']} Score: {alt['processing_score']} (↓{alt['score_improvement']} points better)")
            print(f"  NOVA: {alt['nova_group']} | E-numbers: {alt['e_numbers']}")
