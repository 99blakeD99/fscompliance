"""
MCP Server configuration for FSCompliance.

This module handles MCP-specific configuration settings including
transport protocols, server capabilities, and client connection settings.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TransportType(str, Enum):
    """Supported MCP transport types."""
    STDIO = "stdio"
    SSE = "sse" 
    HTTP = "http"
    WEBSOCKET = "websocket"


class ServerCapabilities(BaseModel):
    """MCP server capabilities configuration."""
    
    # Core capabilities
    tools: bool = Field(True, description="Support for tools/functions")
    resources: bool = Field(True, description="Support for resources")
    prompts: bool = Field(True, description="Support for prompts")
    
    # Advanced capabilities
    logging: bool = Field(True, description="Support for logging")
    sampling: bool = Field(False, description="Support for LLM sampling")
    experimental: Dict[str, bool] = Field(default_factory=dict, description="Experimental features")


class MCPServerConfig(BaseModel):
    """MCP Server configuration settings."""
    
    # Server identity
    name: str = Field("FSCompliance", description="Server name")
    version: str = Field("0.1.0", description="Server version")
    description: str = Field(
        "FSCompliance MCP server for financial regulatory compliance",
        description="Server description"
    )
    
    # Transport configuration
    transport: TransportType = Field(TransportType.STDIO, description="Transport protocol")
    host: str = Field("localhost", description="Host for HTTP/WebSocket transports")
    port: int = Field(8000, description="Port for HTTP/WebSocket transports")
    
    # Server capabilities
    capabilities: ServerCapabilities = Field(default_factory=ServerCapabilities)
    
    # Compliance-specific settings
    default_regulatory_framework: str = Field("FCA_HANDBOOK", description="Default regulatory framework")
    max_analysis_requests: int = Field(100, description="Maximum concurrent analysis requests")
    request_timeout_seconds: int = Field(30, description="Request timeout in seconds")
    
    # Privacy and security
    enable_memory: bool = Field(True, description="Enable long-term memory features")
    anonymize_data: bool = Field(True, description="Anonymize sensitive data in logs")
    require_authentication: bool = Field(False, description="Require client authentication")
    
    # Resource limits
    max_policy_text_length: int = Field(50000, description="Maximum policy text length for analysis")
    max_requirements_returned: int = Field(50, description="Maximum requirements returned per query")
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "FSCOMPLIANCE_MCP_"
        use_enum_values = True


class ClientConnectionConfig(BaseModel):
    """Configuration for client connections."""
    
    # Connection limits
    max_concurrent_connections: int = Field(10, description="Maximum concurrent client connections")
    connection_timeout_seconds: int = Field(60, description="Connection timeout in seconds")
    
    # Rate limiting
    enable_rate_limiting: bool = Field(True, description="Enable rate limiting")
    requests_per_minute: int = Field(60, description="Requests per minute per client")
    burst_limit: int = Field(10, description="Burst request limit")
    
    # Client validation
    allowed_client_names: Optional[List[str]] = Field(None, description="Allowed client names (None = all)")
    require_client_capabilities: List[str] = Field(
        default_factory=list,
        description="Required client capabilities"
    )


def get_mcp_server_config() -> MCPServerConfig:
    """Get MCP server configuration with environment variable support."""
    return MCPServerConfig()


def get_client_connection_config() -> ClientConnectionConfig:
    """Get client connection configuration."""
    return ClientConnectionConfig()