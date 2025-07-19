from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Application
    app_name: str = "Customer Service Bot"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True
    
    # Database
    database_url: str = "postgresql://user:password@localhost/customer_service_bot"
    redis_url: str = "redis://localhost:6379"
    
    # Vector Database
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    
    # LLM Providers
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    venice_ai_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    
    # LLM Provider URLs
    venice_ai_base_url: str = "https://api.venice.ai/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ollama_base_url: str = "http://localhost:11434"
    
    # Security
    secret_key: str = "your-secret-key-change-this-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Logging
    log_level: str = "INFO"
    
    # RAG Configuration
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    max_context_tokens: int = 4000
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()