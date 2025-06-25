"""
Unit tests for FSCompliance MCP Server.
"""

import pytest
from unittest.mock import AsyncMock, patch

from fscompliance.server import FSComplianceServer, get_mcp_server_config
from fscompliance.server.health import health_tracker


class TestFSComplianceServer:
    """Test FSCompliance MCP Server functionality."""

    def test_server_initialization(self):
        """Test server initializes correctly."""
        server = FSComplianceServer("TestServer")
        
        assert server.mcp.name == "TestServer"
        assert server.settings is not None
        
    def test_server_config(self):
        """Test server configuration."""
        config = get_mcp_server_config()
        
        assert config.name == "FSCompliance"
        assert config.version == "0.1.0"
        assert config.capabilities.tools is True
        assert config.capabilities.resources is True
        assert config.capabilities.prompts is True

    def test_health_tracker(self):
        """Test health tracker functionality."""
        initial_connections = health_tracker.connection_count
        
        health_tracker.increment_connections()
        assert health_tracker.connection_count == initial_connections + 1
        
        health_tracker.decrement_connections()
        assert health_tracker.connection_count == initial_connections
        
        health_tracker.record_request(success=True)
        health_tracker.record_request(success=False)
        
        assert health_tracker.total_requests >= 2
        assert health_tracker.failed_requests >= 1
        assert health_tracker.get_error_rate() > 0

    @pytest.mark.asyncio
    async def test_health_check_tool(self):
        """Test health check tool returns valid status."""
        from fscompliance.server.health import health_check_tool
        
        health_status = await health_check_tool()
        
        assert "status" in health_status
        assert "timestamp" in health_status
        assert "uptime_seconds" in health_status
        assert "server" in health_status
        assert "metrics" in health_status
        assert "dependencies" in health_status
        
        assert health_status["server"]["name"] == "FSCompliance MCP Server"
        assert health_status["server"]["version"] == "0.1.0"

    def test_server_get_instance(self):
        """Test getting FastMCP server instance."""
        server = FSComplianceServer()
        mcp_instance = server.get_server()
        
        assert mcp_instance is not None
        assert hasattr(mcp_instance, 'name')

    @pytest.mark.asyncio 
    async def test_server_run_stdio(self):
        """Test server run with stdio transport."""
        server = FSComplianceServer()
        
        # Mock the run_stdio method to avoid actually starting the server
        with patch.object(server.mcp, 'run_stdio', new_callable=AsyncMock) as mock_run:
            await server.run(transport="stdio")
            mock_run.assert_called_once()

    def test_uptime_calculation(self):
        """Test uptime calculation."""
        uptime = health_tracker.get_uptime_seconds()
        assert uptime >= 0
        assert isinstance(uptime, float)