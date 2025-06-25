"""
Unit tests for FSCompliance configuration.
"""

import pytest
from fscompliance.config import Settings, get_settings


class TestSettings:
    """Test configuration settings."""

    def test_default_settings(self):
        """Test default configuration values."""
        settings = Settings()
        
        assert settings.app_name == "FSCompliance"
        assert settings.app_version == "0.1.0"
        assert settings.host == "localhost"
        assert settings.port == 8000
        assert settings.debug is False
        assert settings.memory_enabled is True

    def test_settings_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv("FSCOMPLIANCE_DEBUG", "true")
        monkeypatch.setenv("FSCOMPLIANCE_PORT", "9000")
        monkeypatch.setenv("FSCOMPLIANCE_DEFAULT_LLM", "falcon")
        
        settings = Settings()
        
        assert settings.debug is True
        assert settings.port == 9000
        assert settings.default_llm == "falcon"

    def test_get_settings_cached(self):
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2