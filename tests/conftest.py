"""Pytest configuration and fixtures for component tests."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path so we can import modal_app
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_input():
    """Provide sample REAL input data for testing.
    
    IMPORTANT: This fixture should contain realistic example data
    that matches the expected input format. For actual runs,
    real data from verified sources must be used.
    
    Do NOT use this as a template for generating synthetic data.
    """
    return {
        "source": "test_fixture",
        "parameters": {},
    }
