import os
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

# 1. Load Data
load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Fetch just the countries column
response = supabase.table("food_items").select("product_name, countries_sold").execute()
df = pd.DataFrame(response.data)

print(f"Original Row Count: {len(df)}")

# 2. The Cleaning Function
def clean_country_list(raw_string):
    if not raw_string:
        return []
    
    # Step A: Lowercase everything to make matching easy
    raw_string = raw_string.lower()
    
    # Step B: Remove the annoying 'en:' prefix
    raw_string = raw_string.replace("en:", "")
    
    # Step C: Split by comma
    countries = raw_string.split(',')
    
    # Step D: Strip spaces (e.g. " ireland" -> "ireland")
    clean_countries = [c.strip() for c in countries]
    
    return clean_countries

# 3. Apply the cleaning
# This creates a new column which is a List, not a String
df['country_list'] = df['countries_sold'].apply(clean_country_list)

# 4. Feature Engineering: "Is it sold in Ireland?"
# Look for 'ireland' OR 'ie' in the list
df['sold_in_ireland'] = df['country_list'].apply(lambda x: 'ireland' in x or 'ie' in x)

# 5. Let's look at the results
print("\n--- DATA CHECK ---")
print(df[['countries_sold', 'country_list', 'sold_in_ireland']].head(10))

print("\n--- SUMMARY STATS ---")
print(f"Total Products: {len(df)}")
print(f"Sold in Ireland: {df['sold_in_ireland'].sum()}")
print(f"Not listed as sold in Ireland: {len(df) - df['sold_in_ireland'].sum()}")