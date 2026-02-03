import requests
import json

def get_product_data(barcode):
    # OpenFoodFacts API URL for a specific product
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    headers = {
        'User-Agent': 'UPF_Scanner_Student_Project/1.0 (test@example.com)'
    }
    
    print(f"Checking barcode: {barcode}...")
    response = requests.get(url)
    print(f"API Status Code: {response.status_code}") # 200 means OK, 404 means URL wrong, 403 means Blocked
    
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 1: # 1 means found
            product = data['product']
            
            print(f"--- Product: {product.get('product_name', 'Unknown')} ---")
            print(f"Brand: {product.get('brands', 'Unknown')}")
            
            # The Gold Mine: The raw ingredient text
            print(f"\n[Raw Ingredients]:\n{product.get('ingredients_text', 'NO DATA')}")
            
            # Often have a 'nova_group' already calculated (1-4).
            # Your job will be to see if you can REPLICATE or IMPROVE this logic.
            print(f"\n[Existing NOVA Score]: {product.get('nova_group', 'Not calculated')}")
            
            # Extracting specific nutrient levels
            nutrients = product.get('nutriments', {})
            print(f"\n[Nutrients per 100g]:")
            print(f" - Sugar: {nutrients.get('sugars_100g', '?')}g")
            print(f" - Salt: {nutrients.get('salt_100g', '?')}g")
            
        else:
            print("Product not found.")
    else:
        print("Failed to connect to API.")

# Try it with a real barcode (Go grab a packet of crisps or biscuits from your cupboard!)
# If you don't have one, here is a standard Tayto Cheese & Onion barcode (Ireland):
# Or a generic Coca Cola: 5449000000996
test_barcode = "5000127042858" # Tayto Cheese & Onion 37g
# test_barcode = "5000127042858" # Coke Zero 

get_product_data(test_barcode)