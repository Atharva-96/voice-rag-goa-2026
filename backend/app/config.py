import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    SARVAM_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    # Qdrant Config
    QDRANT_API_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "msmarco_xi_hindi"

    # Models
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    GROQ_MODEL_NAME: str = "groq/compound-mini"

    # Environment
    ENV: str = "development"
    PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
