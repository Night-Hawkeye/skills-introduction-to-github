import pytest

def test_bots_init_imports():
    """Test that the bots package can be imported and exposes PositiveVibesBot."""
    import bots
    assert hasattr(bots, "PositiveVibesBot")
    from bots import PositiveVibesBot
    assert PositiveVibesBot is not None

def test_bots_all_exports():
    """Test that __all__ is correctly defined in bots/__init__.py."""
    import bots
    assert hasattr(bots, "__all__")
    assert "PositiveVibesBot" in bots.__all__
