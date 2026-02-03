import requests
import json

def search_product(query_name):
    # We use the search endpoint instead of the specific product endpoint
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    
    # These parameters tell the API to return JSON data for your search term
    params = {
        'search_terms': query_name,
        'search_simple': 1,
        'action': 'process',
        'json': 1,
        'page_size': 3  # Let's just look at the top 3 results
    }
    
    headers = {
        'User-Agent': 'UPF_Scanner_Student_Project/1.0 (test@example.com)'
    }
    
    print(f"Searching for '{query_name}'...")
    response = requests.get(url, params=params, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        products = data.get('products', [])
        
        if not products:
            print("No products found!")
            return

        print(f"\nFound {len(products)} results. Showing top 3:\n")
        
        for product in products:
            print(f"--- {product.get('product_name', 'Unknown Name')} ---")
            print(f"Barcode: {product.get('code')}")
            print(f"NOVA Score: {product.get('nova_group', 'Unknown')}")
            # This fetches the raw ingredient list
            print(f"Ingredients: {product.get('ingredients_text', 'No ingredients listed')[:100]}...") # First 100 chars
            print("-" * 30)
            
    else:
        print(f"Error: {response.status_code}")

# Try searching for something Irish
search_product("Tayto Cheese and Onion")