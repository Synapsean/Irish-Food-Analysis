"""
Unit tests for the tokenizer module.

Tests the ingredient parsing logic including:
- Basic comma splitting
- Nested parentheses handling
- Synonym normalization
- Allergen warning filtering
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tokenizer import (
    tokenize_ingredients,
    normalize_ingredient,
    parse_ingredient_list,
    DEFAULT_REPLACEMENTS,
)


class TestTokenizeIngredients:
    """Tests for the tokenize_ingredients function."""
    
    def test_simple_comma_separated(self):
        """Should split basic comma-separated ingredients."""
        text = "Water, Salt, Sugar"
        result = tokenize_ingredients(text)
        assert result == ["Water", "Salt", "Sugar"]
    
    def test_empty_string(self):
        """Should return empty list for empty input."""
        assert tokenize_ingredients("") == []
        assert tokenize_ingredients(None) == []
    
    def test_single_ingredient(self):
        """Should handle single ingredient with no commas."""
        result = tokenize_ingredients("Water")
        assert result == ["Water"]
    
    def test_nested_parentheses(self):
        """Should NOT split on commas inside parentheses."""
        text = "Beef (10%), Spices (salt, pepper, paprika), Onion"
        result = tokenize_ingredients(text)
        
        assert len(result) == 3
        assert result[0] == "Beef (10%)"
        assert result[1] == "Spices (salt, pepper, paprika)"
        assert result[2] == "Onion"
    
    def test_complex_nested_ingredients(self):
        """Should handle real-world complex ingredient lists."""
        text = "Water, Vegetable Oil (Rapeseed, Sunflower), Emulsifier (E471, E472e), Salt"
        result = tokenize_ingredients(text)
        
        assert len(result) == 4
        assert "Vegetable Oil (Rapeseed, Sunflower)" in result
        assert "Emulsifier (E471, E472e)" in result
    
    def test_whitespace_trimming(self):
        """Should trim whitespace from each ingredient."""
        text = "  Water  ,  Salt  ,  Sugar  "
        result = tokenize_ingredients(text)
        assert result == ["Water", "Salt", "Sugar"]


class TestNormalizeIngredient:
    """Tests for the normalize_ingredient function."""
    
    def test_lowercase(self):
        """Should convert to lowercase."""
        assert normalize_ingredient("WATER") == "water"
        assert normalize_ingredient("Salt") == "salt"
    
    def test_remove_periods(self):
        """Should remove trailing periods."""
        assert normalize_ingredient("sugar.") == "sugar"
    
    def test_synonym_replacement(self):
        """Should apply synonym mappings."""
        assert normalize_ingredient("flavourings") == "flavouring"
        assert normalize_ingredient("vegetable oils") == "vegetable oil"
        assert normalize_ingredient("glucose syrup") == "sugar"
    
    def test_skip_allergen_warnings(self):
        """Should return empty string for allergen warnings."""
        assert normalize_ingredient("contains milk") == ""
        assert normalize_ingredient("including wheat") == ""
        assert normalize_ingredient("may contain nuts") == ""
        assert normalize_ingredient("Allergen advice: contains gluten") == ""
    
    def test_skip_short_strings(self):
        """Should skip single-character results."""
        assert normalize_ingredient("a") == ""
        assert normalize_ingredient(".") == ""
    
    def test_custom_replacements(self):
        """Should accept custom replacement dictionary."""
        custom = {"h2o": "water", "nacl": "salt"}
        assert normalize_ingredient("H2O", custom) == "water"
        assert normalize_ingredient("NaCl", custom) == "salt"


class TestParseIngredientList:
    """Integration tests for the full parsing pipeline."""
    
    def test_full_pipeline(self):
        """Should tokenize and normalize in one call."""
        text = "WATER, Flavourings, Salt."
        result = parse_ingredient_list(text)
        
        assert "water" in result
        assert "flavouring" in result  # Normalized from "flavourings"
        assert "salt" in result
    
    def test_filters_allergens(self):
        """Should remove allergen warnings from final list."""
        text = "Water, Salt, Contains milk, Sugar"
        result = parse_ingredient_list(text)
        
        assert "water" in result
        assert "salt" in result
        assert "sugar" in result
        assert "contains milk" not in result
    
    def test_handles_complex_real_world(self):
        """Should handle realistic Irish food product ingredients."""
        text = """
        Pork (58%), Water, Rusk (Wheat Flour, Salt), Salt, 
        Spices (White Pepper, Nutmeg, Mace), Dextrose, 
        Preservative (E223), Antioxidant (E301), Contains Sulphites
        """
        result = parse_ingredient_list(text)
        
        # Should have main ingredients
        assert any("pork" in r for r in result)
        assert any("water" in r for r in result)
        
        # Should preserve nested content
        assert any("wheat flour" in r for r in result)
        
        # Should NOT have allergen warning
        assert not any("contains sulphites" in r for r in result)


class TestEdgeCases:
    """Edge case and regression tests."""
    
    def test_multiple_nested_levels(self):
        """Should handle deeply nested parentheses."""
        text = "Oil (Vegetable (Sunflower, Rapeseed)), Salt"
        result = tokenize_ingredients(text)
        # This is a known limitation - may not handle 2+ levels perfectly
        assert len(result) >= 2
    
    def test_unicode_characters(self):
        """Should handle non-ASCII characters."""
        text = "Café, Crème fraîche, Rösti"
        result = tokenize_ingredients(text)
        assert len(result) == 3
    
    def test_numbers_and_percentages(self):
        """Should preserve numbers and percentages."""
        text = "Beef (85%), Water (10%), Salt (5%)"
        result = tokenize_ingredients(text)
        assert "Beef (85%)" in result


# Fixtures for shared test data
@pytest.fixture
def irish_sausage_ingredients():
    """Real Irish sausage ingredient list for testing."""
    return """
    Pork (42%), Water, Rusk (Fortified Wheat Flour (Wheat Flour, 
    Calcium Carbonate, Iron, Niacin, Thiamin), Salt), Pork Fat, 
    Seasoning (Salt, Spices, Dextrose, Herb Extracts, Spice Extracts, 
    Antioxidant: Sodium Ascorbate), Stabiliser: Diphosphates, 
    Preservative: Sodium Metabisulphite. Contains: Wheat, Sulphites.
    """


def test_irish_sausage_parsing(irish_sausage_ingredients):
    """Regression test with real Irish sausage data."""
    result = parse_ingredient_list(irish_sausage_ingredients)
    
    # Should extract main ingredients
    assert any("pork" in r for r in result)
    assert any("water" in r for r in result)
    
    # Should have reasonable count (not over-split)
    assert 5 <= len(result) <= 15
