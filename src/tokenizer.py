"""
Ingredient Tokenizer Module

Handles parsing of complex ingredient lists with nested parentheses.
Uses regex negative lookahead to split on commas only outside brackets.
"""

import re
from typing import List, Dict

# Regex pattern: split on commas NOT inside parentheses
INGREDIENT_PATTERN = r',\s*(?![^()]*\))'

# Synonym mappings for normalization
DEFAULT_REPLACEMENTS: Dict[str, str] = {
    'flavourings': 'flavouring',
    'natural flavourings': 'natural flavouring',
    'spices': 'spice',
    'vegetable oils': 'vegetable oil',
    'sugar syrup': 'sugar',
    'glucose syrup': 'sugar',
    'glucose-fructose syrup': 'sugar',
    'fructose': 'sugar',
}


def tokenize_ingredients(text: str) -> List[str]:
    """
    Split ingredient text into individual ingredients.
    
    Handles nested parentheses like:
    "Water, Beef (10%), Spices (salt, pepper, paprika), Onion"
    
    Args:
        text: Raw ingredient string from food label
        
    Returns:
        List of individual ingredient strings (whitespace-trimmed)
    """
    if not text:
        return []
    
    return [item.strip() for item in re.split(INGREDIENT_PATTERN, text) if item.strip()]


def normalise_ingredient(ingredient: str, replacements: Dict[str, str] = None) -> str:
    """
    Clean and normalise a single ingredient.
    
    - Lowercase
    - Remove trailing periods
    - Apply synonym replacements
    - Skip allergen warnings
    
    Args:
        ingredient: Single ingredient string
        replacements: Optional custom synonym mapping
        
    Returns:
        Normalised ingredient string, or empty string if should be skipped
    """
    if replacements is None:
        replacements = DEFAULT_REPLACEMENTS
    
    # Clean basic formatting
    clean = ingredient.strip().lower().replace('.', '')
    
    # Skip allergen warnings
    skip_prefixes = ('including ', 'contains ', 'contains:', 'may contain ', 'allergen')
    if any(clean.startswith(prefix) for prefix in skip_prefixes):
        return ''
    
    # Apply synonym normalisation
    if clean in replacements:
        clean = replacements[clean]
    
    return clean if len(clean) > 1 else ''


def parse_ingredient_list(text: str, replacements: Dict[str, str] = None) -> List[str]:
    """
    Full pipeline: tokenize and normalise ingredient text.
    
    Args:
        text: Raw ingredient string
        replacements: Optional custom synonym mapping
        
    Returns:
        List of cleaned, normalised ingredients
    """
    tokens = tokenize_ingredients(text)
    normalised = [normalise_ingredient(t, replacements) for t in tokens]
    return [n for n in normalised if n]  # Filter empty strings


# CLI demo when run directly
if __name__ == "__main__":
    import os
    from supabase import create_client, Client
    from dotenv import load_dotenv
    
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    supabase: Client = create_client(url, key)
    
    # Get a product with nested ingredients
    response = supabase.table("food_items") \
        .select("product_name, ingredients_text") \
        .ilike("ingredients_text", "%(%") \
        .limit(1) \
        .execute()
    
    if not response.data:
        print("No complex ingredients found!")
        exit()
    
    data = response.data[0]
    print(f"--- Product: {data['product_name']} ---")
    print(f"RAW TEXT: {data['ingredients_text']}\n")
    
    ingredients = parse_ingredient_list(data['ingredients_text'])
    
    print("--- CLEAN LIST ---")
    for i, ing in enumerate(ingredients, 1):
        print(f"{i}. {ing}")