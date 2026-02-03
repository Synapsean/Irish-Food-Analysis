import os
import requests
import time
from supabase import create_client, Client
from dotenv import load_dotenv

# Load Env Vaiables
load_dotenv()

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")

if not URL or not KEY:
    print("Error: Supabase credentials not found in .env")
    exit(1)

supabase: Client = create_client(URL, KEY)

CATEGORIES = [
    "Biscuits", "Bread", "Yogurt", "Cereal", "Crisps", 
    "Sausages", "Jam", "Soup", "Pizza", "Cheese",
    "Soft Drinks", "Frozen Meals"
]

def harvest_category(category_name, page=1, num_items=50):
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    
    params = {
            'action': 'process',
            'json': 1,
            'page': page,
            'page_size': num_items,
            
            # Filter 1: The Category (e.g., Biscuits)
            'tagtype_0': 'categories',
            'tag_contains_0': 'contains',
            'tag_0': category_name,
            
            # Filter 2: The Country (MUST be Ireland)
            'tagtype_1': 'countries',
            'tag_contains_1': 'contains',
            'tag_1': 'Ireland',
            
            'fields': 'product_name,code,ingredients_text,nova_group,brands,nutriments,origins,link,countries,ecoscore_grade,ecoscore_score,packaging'
        }
    
    headers = {'User-Agent': 'UPF_Student_Project/1.0'}
    
    print(f"Harvesting {category_name}...")
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        return data.get('products', [])
    except Exception as e:
        print(f"Error harvesting {category_name}: {e}")
        return []

def main():
    total_inserted = 0
    
    for cat in CATEGORIES:
        print(f"Starting Category: {cat}")

        for page_num in range(1, 6):
            products = harvest_category(cat, page=page_num)
            if not products:
                print(f"No More Products for {cat} on page {page_num}.")
                break

            batch_data = []
        
            for p in products:
                # Skip items with no barcode or no ingredients
                if not p.get('code') or not p.get('ingredients_text'):
                    continue

                # Clean numeric data (handle empty/None values)
                sugar = p.get('nutriments', {}).get('sugars_100g', 0)
                salt = p.get('nutriments', {}).get('salt_100g', 0)
                fat = p.get('nutriments', {}).get('fat_100g', 0)
                
                # Prepare rows
                row = {
                    'code': p.get('code'),
                    'product_name': p.get('product_name', 'Unknown'),
                    'brand': p.get('brands', 'Unknown'),
                    'nova_group': p.get('nova_group', None),
                    'ingredients_text': p.get('ingredients_text', ''),
                    'sugar_100g': float(sugar) if sugar else 0.0,
                    'salt_100g': float(salt) if salt else 0.0,
                    'fat_100g': float(fat) if fat else 0.0,
                    'countries_sold': p.get('countries', 'Unknown'),
                    'ecoscore_grade': p.get('ecoscore_grade', 'unknown'),
                    'ecoscore_score': float(p.get('ecoscore_score', 0)) if p.get('ecoscore_score') else None,
                    'packaging_tags': p.get('packaging', 'Unknown'),
                    'origin_of_ingredients': p.get('origins', 'Unknown'),
                    'producer_link': p.get('link', ''),
                    'countries_sold': p.get('countries', 'Unknown'),
                    'category_searched': cat
                }
                batch_data.append(row)
            
            # Upsert to Supabase
            if batch_data:
                try:
                    data, count = supabase.table("food_items").upsert(batch_data).execute()
                    print(f" -> Saved {len(batch_data)} items from {cat}")
                    total_inserted += len(batch_data)
                except Exception as e:
                    print(f" -> DB Error on {cat}: {e}")
            
            time.sleep(1) # Please dont ban me
        print(f"Finished {cat}. Taking a chill pill...")
        time.sleep(2) # Pretty please

    print(f"\nJOB COMPLETE. Total items in Database: {total_inserted}")

if __name__ == "__main__":
    main()