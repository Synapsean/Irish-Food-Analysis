import os
import re
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# 1. Get a product with a messy ingredient list
# We specifically ask for one that contains parentheses " ("
response = supabase.table("food_items") \
    .select("product_name, ingredients_text") \
    .ilike("ingredients_text", "%(%") \
    .limit(1) \
    .execute()

if not response.data:
    print("No complex ingredients found!")
    exit()

data = response.data[0]
product = data['product_name']
text = data['ingredients_text']

print(f"--- Product: {product} ---")
print(f"RAW TEXT: {text}\n")

# 2. THE REGEX FIX
# This pattern handles the "Commas inside brackets" problem
pattern = r',\s*(?![^()]*\))'

# Split and clean whitespace
ingredients = [item.strip() for item in re.split(pattern, text)]

print("--- CLEAN LIST ---")
for i, ing in enumerate(ingredients):
    print(f"{i+1}. {ing}")