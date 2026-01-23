import os
import re
import pandas as pd
from collections import Counter # <--- The Counting Tool
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

print("Fetching Irish data...")

# 1. Fetch ALL rows where country contains 'Ireland' or 'ie'
# (We increase limit to 2000 to get them all)
response = supabase.table("food_items") \
    .select("product_name, ingredients_text") \
    .or_("countries_sold.ilike.%Ireland%,countries_sold.ilike.%en:ie%") \
    .neq("ingredients_text", "") \
    .limit(2000) \
    .execute()

products = response.data
print(f"Analysing {len(products)} Irish Products...")

# 2. The Tokenizer (Your Regex)
# We handle "Commas inside brackets"
pattern = r',\s*(?![^()]*\))'
all_ingredients = []

#Common duplicates 
replacements = {
    'flavourings': 'flavouring',
    'natural flavourings': 'natural flavouring',
    'spices': 'spice',
    'vegetable oils': 'vegetable oil',
    'sugar syrup': 'sugar',
    'glucose syrup': 'sugar', # Optional: Grouping all sugars?
}

for p in products:
    text = p['ingredients_text']
    if not text:
        continue
        
    # Split the text
    items = re.split(pattern, text)
    
    # Clean up each item (strip spaces, lowercase, duplicates and allergens)
    for item in items:
            # 1. Clean basic junk
            clean_item = item.strip().lower().replace('.', '')
            
            # 2. Skip Allergy advice (starts with "including", "contains")
            if clean_item.startswith("including ") or clean_item.startswith("contains "):
                continue
                
            # 3. Apply the "Replacements" (Fixing plurals)
            # If the item is in our dictionary, swap it for the better name
            if clean_item in replacements:
                clean_item = replacements[clean_item]
                
            if len(clean_item) > 1:
                all_ingredients.append(clean_item)

# 3. The Analysis
# Count frequency of every ingredient
counts = Counter(all_ingredients)

# 4. Print the Top 20
print("\n--- TOP 20 INGREDIENTS IN IRELAND ---")
df_stats = pd.DataFrame(counts.most_common(20), columns=['Ingredient', 'Count'])
print(df_stats)

import matplotlib.pyplot as plt
import seaborn as sns

# Set the size (Make it wide enough for the text)
plt.figure(figsize=(12, 6))

# Use barplot because you already calculated the counts
sns.barplot(
    data=df_stats, 
    x='Ingredient', 
    y='Count', 
    palette='viridis' # 'viridis' is a nice blue-to-yellow color scheme
)

# Add titles
plt.title('Top 20 Most Common Ingredients in Irish Food Products', fontsize=16)
plt.xlabel('Ingredient', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)

# ROTATE LABELS so they don't crash into each other
plt.xticks(rotation=45, ha='right')

# Fix layout to prevent cutting off the text
plt.tight_layout()

# Save the plot as a high-quality PNG
plt.savefig('irish_food_ingredients.png', dpi=300, bbox_inches='tight')
print("Graph saved to irish_food_ingredients.png")

# Show the graph
print("Opening Graph...")
plt.show()