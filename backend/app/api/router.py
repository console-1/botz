from fastapi import APIRouter
from .endpoints import chat, clients, knowledge_base, analytics, health

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(clients.router, prefix="/clients", tags=["clients"])
api_router.include_router(knowledge_base.router, prefix="/knowledge-base", tags=["knowledge-base"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])