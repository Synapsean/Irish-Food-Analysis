"""
Unit tests for the recommender module.

Tests additive detection, processing score calculation, and alternative finding.
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from recommender import (
    count_e_numbers,
    find_concerning_additives,
    calculate_processing_score,
    get_nova_label,
    get_processing_badge,
    find_healthier_alternatives,
    get_product_analysis,
)


class TestENumberCounting:
    """Tests for E-number detection."""
    
    def test_simple_e_numbers(self):
        """Should count standard E-numbers."""
        text = "Water, Sugar, E150, E330, Salt"
        assert count_e_numbers(text) == 2
    
    def test_e_number_with_letter(self):
        """Should count E-numbers with letter suffix."""
        text = "Flour, E471a, E472b, Yeast"
        assert count_e_numbers(text) == 2
    
    def test_no_e_numbers(self):
        """Should return 0 for clean ingredients."""
        text = "Water, Flour, Salt, Sugar"
        assert count_e_numbers(text) == 0
    
    def test_empty_string(self):
        """Should handle empty input."""
        assert count_e_numbers("") == 0
        assert count_e_numbers(None) == 0
    
    def test_complex_ingredients(self):
        """Should count E-numbers in complex ingredient lists."""
        text = "Wheat flour, water, E322 (from soya), vegetable oils (E471, E472e), salt"
        assert count_e_numbers(text) == 3


class TestConcerningAdditives:
    """Tests for concerning additive detection."""
    
    def test_finds_aspartame(self):
        """Should detect artificial sweeteners."""
        text = "Carbonated water, aspartame, phosphoric acid"
        additives = find_concerning_additives(text)
        found_names = [name for name, _ in additives]
        assert "aspartame" in found_names
        assert "phosphoric acid" in found_names
    
    def test_case_insensitive(self):
        """Should work regardless of case."""
        text = "ASPARTAME, Monosodium Glutamate"
        additives = find_concerning_additives(text)
        found_names = [name for name, _ in additives]
        assert "aspartame" in found_names
        assert "monosodium glutamate" in found_names
    
    def test_no_additives_clean_product(self):
        """Should return empty for clean products."""
        text = "Whole wheat flour, water, sea salt, olive oil"
        assert find_concerning_additives(text) == []
    
    def test_empty_string(self):
        """Should handle empty input."""
        assert find_concerning_additives("") == []


class TestProcessingScore:
    """Tests for processing score calculation."""
    
    def test_nova_1_low_score(self):
        """NOVA 1 products should have low scores."""
        row = pd.Series({
            'nova_group': 1,
            'ingredients_text': "Water, apple, sugar"
        })
        score = calculate_processing_score(row)
        assert score <= 30
    
    def test_nova_4_high_score(self):
        """NOVA 4 products should have high scores."""
        row = pd.Series({
            'nova_group': 4,
            'ingredients_text': "Water, aspartame, E150, E330, phosphoric acid"
        })
        score = calculate_processing_score(row)
        assert score >= 50
    
    def test_score_capped_at_100(self):
        """Score should never exceed 100."""
        row = pd.Series({
            'nova_group': 4,
            'ingredients_text': "E100, E200, E300, E400, E500, aspartame, phosphoric acid, monosodium glutamate"
        })
        score = calculate_processing_score(row)
        assert score <= 100


class TestNovaLabel:
    """Tests for NOVA label generation."""
    
    def test_all_nova_groups(self):
        """Should return correct labels for all NOVA groups."""
        assert get_nova_label(1) == "Unprocessed/Minimal"
        assert get_nova_label(2) == "Processed Culinary"
        assert get_nova_label(3) == "Processed"
        assert get_nova_label(4) == "Ultra-Processed"
    
    def test_unknown_nova(self):
        """Should handle unknown values."""
        assert get_nova_label(None) == "Unknown"
        assert get_nova_label(5) == "Unknown"
        assert get_nova_label("foo") == "Unknown"


class TestProcessingBadge:
    """Tests for processing badge generation."""
    
    def test_low_score_green(self):
        """Low scores should get green badge."""
        badge, label = get_processing_badge(15)
        assert badge == "🟢"
        assert "Minimal" in label
    
    def test_high_score_red(self):
        """High scores should get red badge."""
        badge, label = get_processing_badge(80)
        assert badge == "🔴"
        assert "Highly" in label


class TestFindAlternatives:
    """Tests for alternative finding."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample product dataframe."""
        return pd.DataFrame([
            {
                'product_name': 'UPF Cola',
                'category_searched': 'Soft Drinks',
                'nova_group': 4,
                'ingredients_text': 'Water, sugar, aspartame, E150, phosphoric acid',
                'brand': 'Brand A',
                'sugar_100g': 10.0,
                'salt_100g': 0.1,
            },
            {
                'product_name': 'Sparkling Water',
                'category_searched': 'Soft Drinks',
                'nova_group': 1,
                'ingredients_text': 'Carbonated natural mineral water',
                'brand': 'Brand B',
                'sugar_100g': 0.0,
                'salt_100g': 0.0,
            },
            {
                'product_name': 'Fruit Juice',
                'category_searched': 'Soft Drinks',
                'nova_group': 2,
                'ingredients_text': 'Apple juice, water',
                'brand': 'Brand C',
                'sugar_100g': 8.0,
                'salt_100g': 0.0,
            },
        ])
    
    def test_finds_healthier_in_category(self, sample_df):
        """Should find less processed alternatives in same category."""
        upf_product = sample_df.iloc[0].to_dict()
        alternatives = find_healthier_alternatives(upf_product, sample_df, top_n=2)
        
        assert len(alternatives) >= 1
        # Sparkling water should be suggested (NOVA 1)
        alt_names = [a['product_name'] for a in alternatives]
        assert 'Sparkling Water' in alt_names
    
    def test_excludes_self(self, sample_df):
        """Should not suggest the input product as alternative."""
        product = sample_df.iloc[0].to_dict()
        alternatives = find_healthier_alternatives(product, sample_df)
        
        alt_names = [a['product_name'] for a in alternatives]
        assert product['product_name'] not in alt_names


class TestProductAnalysis:
    """Tests for full product analysis."""
    
    def test_returns_all_fields(self):
        """Should return all expected analysis fields."""
        product = {
            'product_name': 'Test Product',
            'brand': 'Test Brand',
            'category_searched': 'Test Category',
            'nova_group': 3,
            'ingredients_text': 'Water, sugar, salt, E150',
            'sugar_100g': 5.0,
            'salt_100g': 0.5,
        }
        analysis = get_product_analysis(product)
        
        assert 'processing_score' in analysis
        assert 'processing_badge' in analysis
        assert 'e_numbers' in analysis
        assert 'concerning_additives' in analysis
        assert 'suggestions' in analysis
