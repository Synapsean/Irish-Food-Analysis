"""
Data Quality Report for Irish Food Analysis
===========================================

Generates a comprehensive data quality report for the food products database.
Checks for missing data, outliers, and data quality issues.

Usage:
    python scripts/data_quality_report.py
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from supabase_utils import load_data_from_supabase


def generate_quality_report():
    """Generate comprehensive data quality report."""
    
    print("="*80)
    print("IRISH FOOD ANALYSIS - DATA QUALITY REPORT")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Load data
    print("Loading data from Supabase...")
    df = load_data_from_supabase()
    
    if df.empty:
        print("❌ No data found in database!")
        return
    
    print(f"✓ Loaded {len(df)} products\n")
    
    # =========================================================================
    # SECTION 1: Basic Statistics
    # =========================================================================
    print("="*80)
    print("1. DATASET OVERVIEW")
    print("="*80)
    
    print(f"Total products: {len(df)}")
    print(f"Unique product codes: {df['code'].nunique()}")
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}" if 'created_at' in df else "N/A")
    
    # Categories
    if 'main_category' in df:
        print(f"\nCategories: {df['main_category'].nunique()}")
        print("\nTop 10 categories:")
        print(df['main_category'].value_counts().head(10).to_string())
    
    # NOVA distribution
    if 'nova_group' in df:
        print(f"\nNOVA Group Distribution:")
        nova_counts = df['nova_group'].value_counts().sort_index()
        for nova, count in nova_counts.items():
            pct = (count / len(df)) * 100
            print(f"  NOVA {nova}: {count:>5} products ({pct:>5.1f}%)")
    
    # =========================================================================
    # SECTION 2: Missing Data Analysis
    # =========================================================================
    print("\n" + "="*80)
    print("2. MISSING DATA ANALYSIS")
    print("="*80)
    
    # Critical fields
    critical_fields = ['product_name', 'code', 'ingredients_text', 'nova_group']
    
    print("\nCritical Fields:")
    for field in critical_fields:
        if field in df:
            missing = df[field].isna().sum()
            pct = (missing / len(df)) * 100
            status = "✓ OK" if missing == 0 else f"⚠️  {missing} missing ({pct:.1f}%)"
            print(f"  {field:.<30} {status}")
    
    # Nutrient fields
    nutrient_fields = {
        'salt_100g': 'Salt (per 100g)',
        'sugars_100g': 'Sugars (per 100g)',
        'fat_100g': 'Fat (per 100g)',
        'saturated_fat_100g': 'Saturated Fat (per 100g)',
        'energy_100g': 'Calories/Energy (per 100g)'
    }
    
    print("\nNutrient Data Completeness:")
    for field, label in nutrient_fields.items():
        if field in df:
            present = df[field].notna().sum()
            pct = (present / len(df)) * 100
            print(f"  {label:.<35} {present:>5} / {len(df)} ({pct:>5.1f}%)")
    
    # =========================================================================
    # SECTION 3: Data Quality Issues
    # =========================================================================
    print("\n" + "="*80)
    print("3. DATA QUALITY ISSUES")
    print("="*80)
    
    issues = []
    
    # Duplicates
    duplicate_codes = df[df.duplicated(subset=['code'], keep=False)]
    if len(duplicate_codes) > 0:
        issues.append(f"⚠️  {len(duplicate_codes)} duplicate product codes found")
    
    # Negative nutrients
    for col in ['salt_100g', 'sugars_100g', 'fat_100g']:
        if col in df:
            negative = (df[col] < 0).sum()
            if negative > 0:
                issues.append(f"⚠️  {negative} products with negative {col}")
    
    # Unrealistic nutrient values (e.g., salt > 100g per 100g)
    if 'salt_100g' in df:
        extreme_salt = (df['salt_100g'] > 100).sum()
        if extreme_salt > 0:
            issues.append(f"⚠️  {extreme_salt} products with salt > 100g/100g (impossible)")
    
    if 'sugars_100g' in df:
        extreme_sugar = (df['sugars_100g'] > 100).sum()
        if extreme_sugar > 0:
            issues.append(f"⚠️  {extreme_sugar} products with sugar > 100g/100g")
    
    # Missing ingredients despite having NOVA classification
    if 'nova_group' in df and 'ingredients_text' in df:
        nova_no_ingredients = df[df['nova_group'].notna() & df['ingredients_text'].isna()]
        if len(nova_no_ingredients) > 0:
            issues.append(f"⚠️  {len(nova_no_ingredients)} products have NOVA but no ingredients")
    
    # Empty ingredient lists
    if 'ingredients_text' in df:
        empty_ingredients = df[df['ingredients_text'].str.strip() == '']
        if len(empty_ingredients) > 0:
            issues.append(f"⚠️  {len(empty_ingredients)} products with empty ingredient lists")
    
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  ✓ No major data quality issues detected")
    
    # =========================================================================
    # SECTION 4: Nutrient Statistics
    # =========================================================================
    print("\n" + "="*80)
    print("4. NUTRIENT STATISTICS")
    print("="*80)
    
    nutrient_cols = ['salt_100g', 'sugars_100g', 'fat_100g', 'saturated_fat_100g']
    
    for col in nutrient_cols:
        if col in df and df[col].notna().sum() > 0:
            print(f"\n{col.replace('_', ' ').title()}:")
            print(f"  Mean:   {df[col].mean():.2f}g")
            print(f"  Median: {df[col].median():.2f}g")
            print(f"  Min:    {df[col].min():.2f}g")
            print(f"  Max:    {df[col].max():.2f}g")
            print(f"  Std:    {df[col].std():.2f}g")
    
    # =========================================================================
    # SECTION 5: Recommendations
    # =========================================================================
    print("\n" + "="*80)
    print("5. RECOMMENDATIONS")
    print("="*80)
    
    recommendations = []
    
    # Missing nutrients
    for field, label in nutrient_fields.items():
        if field in df:
            missing_pct = (df[field].isna().sum() / len(df)) * 100
            if missing_pct > 50:
                recommendations.append(f"🔄 {label}: {missing_pct:.0f}% missing - consider data enrichment")
    
    # Missing ingredients
    if 'ingredients_text' in df:
        missing_ingredients = (df['ingredients_text'].isna().sum() / len(df)) * 100
        if missing_ingredients > 30:
            recommendations.append(f"🔄 Ingredients: {missing_ingredients:.0f}% missing - critical for analysis")
    
    # Suggest categories to expand
    if 'main_category' in df:
        small_categories = df['main_category'].value_counts()[df['main_category'].value_counts() < 50]
        if len(small_categories) > 0:
            recommendations.append(f"📊 {len(small_categories)} categories have < 50 products - consider harvesting more")
    
    if recommendations:
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("  ✓ Data quality is good - no immediate actions needed")
    
    # =========================================================================
    # Save Report
    # =========================================================================
    print("\n" + "="*80)
    
    report_dir = Path(__file__).parent.parent / 'outputs' / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = report_dir / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    print(f"📄 Report saved to: {report_file}")
    print("="*80)


# Helper function to load data (fallback if not using Supabase)
def load_data_from_supabase():
    """Load data from Supabase or local CSV fallback."""
    try:
        from dotenv import load_dotenv
        import os
        from supabase import create_client
        
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if url and key:
            supabase = create_client(url, key)
            response = supabase.table("food_products").select("*").execute()
            return pd.DataFrame(response.data)
    except Exception as e:
        print(f"Could not load from Supabase: {e}")
    
    # Fallback: Try to load from local CSV
    csv_path = Path(__file__).parent.parent / 'outputs' / 'food_products.csv'
    if csv_path.exists():
        print(f"Loading from local CSV: {csv_path}")
        return pd.read_csv(csv_path)
    
    return pd.DataFrame()


if __name__ == "__main__":
    generate_quality_report()
