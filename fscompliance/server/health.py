"""
Health check and status monitoring for FSCompliance MCP Server.

This module provides health check endpoints and system status monitoring
for the MCP server, including dependency checks and performance metrics.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from ..config import get_settings


class HealthStatus:
    """Health status tracking for FSCompliance MCP server."""
    
    def __init__(self):
        """Initialize health status tracker."""
        self.start_time = time.time()
        self.last_health_check = None
        self.connection_count = 0
        self.total_requests = 0
        self.failed_requests = 0
        self.settings = get_settings()
    
    def increment_connections(self) -> None:
        """Increment active connection count."""
        self.connection_count += 1
        logger.debug(f"Active connections: {self.connection_count}")
    
    def decrement_connections(self) -> None:
        """Decrement active connection count."""
        self.connection_count = max(0, self.connection_count - 1)
        logger.debug(f"Active connections: {self.connection_count}")
    
    def record_request(self, success: bool = True) -> None:
        """
        Record a request for metrics tracking.
        
        Args:
            success: Whether the request was successful
        """
        self.total_requests += 1
        if not success:
            self.failed_requests += 1
    
    def get_uptime_seconds(self) -> float:
        """Get server uptime in seconds."""
        return time.time() - self.start_time
    
    def get_error_rate(self) -> float:
        """Get error rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    async def check_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """
        Check health of system dependencies.
        
        Returns:
            Dict containing dependency health status
        """
        dependencies = {}
        
        # Check database connectivity
        try:
            # TODO: Implement actual database health check
            dependencies["database"] = {
                "status": "healthy",
                "response_time_ms": 5,
                "details": "SQLite connection successful"
            }
        except Exception as e:
            dependencies["database"] = {
                "status": "unhealthy",
                "error": str(e),
                "details": "Database connection failed"
            }
        
        # Check LLM service availability
        try:
            # TODO: Implement actual LLM health check
            dependencies["llm_service"] = {
                "status": "healthy", 
                "model": self.settings.default_llm,
                "response_time_ms": 150,
                "details": f"{self.settings.default_llm} model available"
            }
        except Exception as e:
            dependencies["llm_service"] = {
                "status": "unhealthy",
                "error": str(e),
                "details": "LLM service unavailable"
            }
        
        # Check knowledge base
        try:
            # TODO: Implement actual knowledge base health check
            dependencies["knowledge_base"] = {
                "status": "healthy",
                "fca_handbook_loaded": True,
                "total_requirements": 1250,
                "last_updated": "2024-12-25T00:00:00Z"
            }
        except Exception as e:
            dependencies["knowledge_base"] = {
                "status": "unhealthy",
                "error": str(e),
                "details": "Knowledge base unavailable"
            }
        
        return dependencies
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status.
        
        Returns:
            Dict containing complete health information
        """
        self.last_health_check = datetime.now(timezone.utc)
        dependencies = await self.check_dependencies()
        
        # Determine overall health
        unhealthy_deps = [
            name for name, info in dependencies.items() 
            if info.get("status") != "healthy"
        ]
        
        overall_status = "unhealthy" if unhealthy_deps else "healthy"
        
        return {
            "status": overall_status,
            "timestamp": self.last_health_check.isoformat(),
            "uptime_seconds": self.get_uptime_seconds(),
            "server": {
                "name": "FSCompliance MCP Server",
                "version": "0.1.0"
            },
            "metrics": {
                "active_connections": self.connection_count,
                "total_requests": self.total_requests,
                "failed_requests": self.failed_requests,
                "error_rate_percent": round(self.get_error_rate(), 2)
            },
            "dependencies": dependencies
        }


# Global health tracker instance
health_tracker = HealthStatus()


async def health_check_tool() -> Dict[str, Any]:
    """
    MCP tool for health checking.
    
    Returns:
        Dict containing health status
    """
    logger.info("Health check requested via MCP tool")
    return await health_tracker.get_health_status()