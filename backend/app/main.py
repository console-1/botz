from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import time
import uuid
from contextlib import asynccontextmanager

from .core.config import settings
from .core.database import engine, Base
from .api.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    """
    # Startup
    print("Starting up Customer Service Bot API...")
    
    # Create database tables
    Base.metadata.create_all(bind=engine)
    
    # Initialize vector database collections if needed
    # This could be moved to a separate initialization script
    
    yield
    
    # Shutdown
    print("Shutting down Customer Service Bot API...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="A generic customer service bot with hot-swappable knowledge bases",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for production
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]  # Configure this properly for production
    )


# Request tracking middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Add request processing time and request ID to response headers
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    return response


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Not found",
            "request_id": getattr(request.state, "request_id", None)
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": getattr(request.state, "request_id", None)
        }
    )


# Include API routes
app.include_router(api_router, prefix="/api/v1")


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with basic API information
    """
    return {
        "message": "Customer Service Bot API",
        "version": settings.app_version,
        "environment": settings.environment,
        "docs_url": "/docs" if settings.debug else None,
        "health_check": "/api/v1/health"
    }


# Widget embedding endpoint
@app.get("/widget.js")
async def widget_script():
    """
    Serve the embeddable widget script
    """
    # This would serve the compiled JavaScript widget
    # For now, return a placeholder
    return JSONResponse(
        content={"message": "Widget script endpoint - to be implemented"},
        media_type="application/javascript"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )