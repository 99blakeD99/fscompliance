"""
FSCompliance configuration settings.

This module handles all configuration for the FSCompliance service,
including environment variables, defaults, and validation.
"""

from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """FSCompliance configuration settings."""
    
    # Application settings
    app_name: str = "FSCompliance"
    app_version: str = "0.1.0"
    debug: bool = Field(False, env="FSCOMPLIANCE_DEBUG")
    
    # Server settings
    host: str = Field("localhost", env="FSCOMPLIANCE_HOST")
    port: int = Field(8000, env="FSCOMPLIANCE_PORT")
    
    # Database settings
    database_url: str = Field("sqlite:///fscompliance.db", env="FSCOMPLIANCE_DB_URL")
    
    # LLM settings
    default_llm: str = Field("llama3", env="FSCOMPLIANCE_DEFAULT_LLM")
    llm_api_key: Optional[str] = Field(None, env="FSCOMPLIANCE_LLM_API_KEY")
    
    # Privacy and memory settings
    memory_enabled: bool = Field(True, env="FSCOMPLIANCE_MEMORY_ENABLED")
    anonymize_data: bool = Field(True, env="FSCOMPLIANCE_ANONYMIZE_DATA")
    
    # Regulatory data settings
    fca_handbook_path: Optional[str] = Field(None, env="FSCOMPLIANCE_FCA_HANDBOOK_PATH")
    knowledge_graph_path: str = Field("./knowledge_graphs", env="FSCOMPLIANCE_KG_PATH")
    
    # Security settings
    secret_key: str = Field("dev-secret-key-change-in-production", env="FSCOMPLIANCE_SECRET_KEY")
    access_token_expire_minutes: int = Field(30, env="FSCOMPLIANCE_TOKEN_EXPIRE_MINUTES")
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()