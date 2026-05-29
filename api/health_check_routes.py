"""Health check endpoints for Kubernetes probes and monitoring."""

import os
import time
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError


health_check_router = APIRouter(prefix="/health", tags=["Health"])


class HealthCheckState:
    """Encapsulates health check state."""

    def __init__(self):
        self.start_time = time.time()
        self.startup_complete = False

    def mark_startup_complete(self):
        """Call this after application initialization is complete."""
        self.startup_complete = True


# Initialize health check state
_health_state = HealthCheckState()


def mark_startup_complete():
    """Call this after application initialization is complete."""
    _health_state.mark_startup_complete()


def check_mongodb() -> Dict[str, Any]:
    """Check MongoDB connectivity."""
    try:
        connection_string = os.environ.get("MONGODB_URI")
        if not connection_string:
            return {
                "status": "unhealthy",
                "message": "MongoDB connection string not configured",
            }

        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        # Ping the database
        client.admin.command("ping")
        client.close()

        return {"status": "healthy"}
    except (ConnectionFailure, ServerSelectionTimeoutError):
        return {
            "status": "unhealthy",
            "message": "MongoDB connection failed",
        }


def check_azure_openai() -> Dict[str, Any]:
    """Check Azure OpenAI configuration."""
    api_key = os.environ.get("DKS_AZURE_OPENAI_API_KEY")
    api_base = os.environ.get("DKS_AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("DKS_AZURE_OPENAI_API_VERSION")

    if not all([api_key, api_base, api_version]):
        return {
            "status": "unhealthy",
            "message": "Azure OpenAI configuration incomplete",
        }

    return {"status": "healthy"}


@health_check_router.get("/")
async def health_check():
    """
    Basic health check endpoint.
    Returns 200 if the service is running.
    """
    return {"status": "OK"}


@health_check_router.get("/live")
async def liveness_probe():
    """
    Kubernetes liveness probe.
    Checks if the application is running and responsive.
    Returns 200 if alive, 503 if dead.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": round(time.time() - _health_state.start_time, 2),
        },
    )


@health_check_router.get("/ready")
async def readiness_probe():
    """
    Kubernetes readiness probe.
    Checks if the application is ready to accept traffic.
    Validates critical dependencies (MongoDB, Azure OpenAI).
    Returns 200 if ready, 503 if not ready.
    """
    checks = {
        "mongodb": check_mongodb(),
        "azure_openai": check_azure_openai(),
    }

    # Determine overall health
    # MongoDB and Azure OpenAI are critical
    is_healthy = (
        checks["mongodb"]["status"] == "healthy"
        and checks["azure_openai"]["status"] == "healthy"
    )

    response_code = (
        status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        status_code=response_code,
        content={
            "status": "healthy" if is_healthy else "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        },
    )


@health_check_router.get("/startup")
async def startup_probe():
    """
    Kubernetes startup probe.
    Checks if the application has completed initialization.
    Returns 200 when startup is complete, 503 otherwise.
    """
    if not _health_state.startup_complete:
        # Auto-mark as complete after 10 seconds (reduced from 30 for faster startup)
        # This handles cases where lifespan event hasn't completed yet
        if time.time() - _health_state.start_time > 10:
            mark_startup_complete()

    response_code = (
        status.HTTP_200_OK
        if _health_state.startup_complete
        else status.HTTP_503_SERVICE_UNAVAILABLE
    )

    return JSONResponse(
        status_code=response_code,
        content={
            "status": "ready" if _health_state.startup_complete else "starting",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": round(time.time() - _health_state.start_time, 2),
            "startup_complete": _health_state.startup_complete,
        },
    )
