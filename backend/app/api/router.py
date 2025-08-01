from fastapi import APIRouter
from .endpoints import (
    chat, clients, knowledge_base, analytics, health, vectors, search, 
    versioning, llm, chat_complete, admin, onboarding, gdpr
)

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(chat_complete.router, prefix="/chat/v2", tags=["chat-complete"])
api_router.include_router(clients.router, prefix="/clients", tags=["clients"])
api_router.include_router(knowledge_base.router, prefix="/knowledge-base", tags=["knowledge-base"])
api_router.include_router(vectors.router, prefix="/vectors", tags=["vectors"])
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(versioning.router, prefix="/versioning", tags=["versioning"])
api_router.include_router(llm.router, prefix="/llm", tags=["llm"])
api_router.include_router(analytics.router, prefix="/analytics", tags=["analytics"])
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
api_router.include_router(onboarding.router, prefix="/onboarding", tags=["onboarding"])
api_router.include_router(gdpr.router, prefix="/gdpr", tags=["gdpr"])