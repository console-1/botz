from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
from ...core.database import get_db
from ...core.vector_db import vector_db
from ...core.redis import redis_client
from ...core.config import settings

router = APIRouter()


@router.get("/", response_model=Dict[str, Any])
async def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint to verify all system components are working
    """
    health_status = {
        "status": "healthy",
        "timestamp": "2025-07-14T00:00:00Z",
        "version": settings.app_version,
        "environment": settings.environment,
        "components": {}
    }
    
    # Check database connectivity
    try:
        db.execute("SELECT 1")
        health_status["components"]["database"] = {"status": "healthy", "type": "postgresql"}
    except Exception as e:
        health_status["components"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check Redis connectivity
    try:
        redis_client.ping()
        health_status["components"]["redis"] = {"status": "healthy", "type": "redis"}
    except Exception as e:
        health_status["components"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check Qdrant connectivity
    try:
        collections = vector_db.list_collections()
        health_status["components"]["vector_db"] = {
            "status": "healthy", 
            "type": "qdrant",
            "collections_count": len(collections)
        }
    except Exception as e:
        health_status["components"]["vector_db"] = {"status": "unhealthy", "error": str(e)}
        health_status["status"] = "unhealthy"
    
    # Check LLM provider access (basic check)
    llm_providers = []
    if settings.openai_api_key:
        llm_providers.append("openai")
    if settings.anthropic_api_key:
        llm_providers.append("anthropic")
    if settings.mistral_api_key:
        llm_providers.append("mistral")
    
    health_status["components"]["llm_providers"] = {
        "status": "configured" if llm_providers else "not_configured",
        "providers": llm_providers
    }
    
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status


@router.get("/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes deployments
    """
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """
    Liveness check for Kubernetes deployments
    """
    return {"status": "alive"}