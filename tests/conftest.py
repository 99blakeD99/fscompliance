"""
Pytest configuration and fixtures for FSCompliance tests.
"""

import pytest
from fscompliance.config import Settings


@pytest.fixture
def test_settings():
    """Provide test configuration settings."""
    return Settings(
        debug=True,
        database_url="sqlite:///:memory:",
        memory_enabled=False,
        secret_key="test-secret-key",
    )