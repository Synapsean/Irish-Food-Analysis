"""
Database Update Script for Irish Food Analysis
==============================================

This script refreshes the Supabase database with latest data from OpenFoodFacts.
Run monthly to keep product information current.

Usage:
    python scripts/update_database.py
    python scripts/update_database.py --categories "Biscuits,Cereal"  # Specific categories only
    python scripts/update_database.py --full-refresh  # Delete and rebuild entire database
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harvester import harvest_category, CATEGORIES


def update_database(categories=None, full_refresh=False):
    """
    Update database with latest OpenFoodFacts data.
    
    Parameters
    ----------
    categories : list, optional
        Specific categories to update. If None, updates all categories.
    full_refresh : bool, default False
        If True, performs full database refresh (slower but more thorough)
    
    Returns
    -------
    dict
        Summary statistics of the update
    """
    if categories is None:
        categories = CATEGORIES
    
    print("="*70)
    print("Irish Food Database Update")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Categories to update: {len(categories)}")
    print(f"Full refresh mode: {full_refresh}")
    print()
    
    total_products = 0
    category_stats = {}
    
    for category in categories:
        print(f"\n{'='*70}")
        print(f"Processing: {category}")
        print(f"{'='*70}")
        
        # Harvest multiple pages to get more products
        # OpenFoodFacts can return 50-100 items per page depending on filters
        category_count = 0
        
        for page in range(1, 6):  # Get up to 5 pages per category
            print(f"  Page {page}/5...", end=" ")
            
            try:
                products = harvest_category(category, page=page, num_items=50)
                
                if products == 0:
                    print("No more products found")
                    break
                
                category_count += products
                print(f"{products} products harvested")
                
            except Exception as e:
                print(f"Error on page {page}: {e}")
                break
        
        category_stats[category] = category_count
        total_products += category_count
        
        print(f"  ✓ Total for {category}: {category_count} products")
    
    # Print summary
    print("\n" + "="*70)
    print("UPDATE COMPLETE")
    print("="*70)
    print(f"\nTotal products processed: {total_products}")
    print(f"\nBreakdown by category:")
    
    for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category:.<30} {count:>4} products")
    
    print(f"\nDatabase updated successfully!")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return {
        'total_products': total_products,
        'categories': category_stats,
        'timestamp': datetime.now().isoformat()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Update Irish Food Analysis database from OpenFoodFacts"
    )
    
    parser.add_argument(
        '--categories',
        type=str,
        help='Comma-separated list of categories to update (e.g., "Biscuits,Cereal")'
    )
    
    parser.add_argument(
        '--full-refresh',
        action='store_true',
        help='Perform full database refresh (slower but more thorough)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be updated without actually updating database'
    )
    
    args = parser.parse_args()
    
    # Parse categories if provided
    categories = None
    if args.categories:
        categories = [c.strip() for c in args.categories.split(',')]
        
        # Validate categories
        invalid = [c for c in categories if c not in CATEGORIES]
        if invalid:
            print(f"Error: Invalid categories: {invalid}")
            print(f"Valid categories: {', '.join(CATEGORIES)}")
            sys.exit(1)
    
    if args.dry_run:
        print("DRY RUN MODE - No database changes will be made")
        print(f"Would update categories: {categories or CATEGORIES}")
        sys.exit(0)
    
    # Confirm full refresh
    if args.full_refresh:
        response = input("⚠️  Full refresh will delete existing data. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)
    
    # Run update
    try:
        stats = update_database(categories=categories, full_refresh=args.full_refresh)
        
        # Save update log
        log_dir = Path(__file__).parent.parent / 'outputs' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"database_update_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(log_file, 'w') as f:
            f.write(f"Database Update Log\n")
            f.write(f"{'='*70}\n")
            f.write(f"Timestamp: {stats['timestamp']}\n")
            f.write(f"Total products: {stats['total_products']}\n\n")
            f.write(f"Categories:\n")
            for cat, count in stats['categories'].items():
                f.write(f"  {cat}: {count}\n")
        
        print(f"\nLog saved to: {log_file}")
        
    except Exception as e:
        print(f"\n❌ Error during update: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
