# RAG_Core/config/settings.py (UPDATED WITH NGROK URL)

from typing import Optional

try:
    from pydantic_settings import BaseSettings

    V2 = True
except Exception:
    from pydantic import BaseSettings

    V2 = False


class Settings(BaseSettings):
    """App settings"""

    # ===== Milvus =====
    MILVUS_HOST: str = "milvus"
    MILVUS_PORT: str = "19530"
    DOCUMENT_COLLECTION: str = "document_chunks"
    FAQ_COLLECTION: str = "faq_embeddings"
    DOCUMENT_URLS_COLLECTION: str = "document_urls"  # NEW

    # ===== Ollama / LLM =====
    OLLAMA_URL: str = "http://ollama:11434"
    LLM_MODEL: str = "gpt-oss:20b"
    OLLAMA_BASE_URL: Optional[str] = None

    # ===== Embedding =====
    EMBEDDING_MODEL: str = "keepitreal/vietnamese-sbert"
    EMBEDDING_DIM: int = 768

    # ===== Search / RAG =====
    SIMILARITY_THRESHOLD: float = 0.5
    TOP_K: int = 20
    MAX_ITERATIONS: int = 5

    # ===== FAQ OPTIMIZATION SETTINGS =====
    FAQ_VECTOR_THRESHOLD: float = 0.5
    FAQ_TOP_K: int = 10
    FAQ_RERANK_THRESHOLD: float = 0.6
    FAQ_QUESTION_WEIGHT: float = 0.5
    FAQ_QA_WEIGHT: float = 0.3
    FAQ_ANSWER_WEIGHT: float = 0.2
    FAQ_CONSISTENCY_BONUS: float = 1.1
    FAQ_CONSISTENCY_THRESHOLD: float = 0.75

    # ===== Document Grader Settings =====
    DOCUMENT_RERANK_THRESHOLD: float = 0.7

    # ===== NEW: DOCUMENT URL SETTINGS =====
    # Internal MinIO URL (used by services)
    MINIO_INTERNAL_URL: str = "http://localhost:9000"

    # Public ngrok URL (for user-facing responses)
    # Example: "https://b1234567.ngrok-free.app"
    NGROK_PUBLIC_URL: Optional[str] = "http://124.158.6.101:9000"

    # Enable URL replacement
    ENABLE_URL_REPLACEMENT: bool = True

    # URL formatting style
    URL_FORMAT_STYLE: str = "detailed"  # simple | detailed | markdown | html

    # Maximum URLs to show in footer
    MAX_REFERENCE_URLS: int = 5
    COHERE_API_KEY: str = "e756puaI9nSZlT967qamaJ7eUdeXh2neu8dy6lGF"
    # ===== Contact =====
    SUPPORT_PHONE: str = "Phòng vận hành 0904540490 - Phòng kinh doanh:0914616081"

    # ===== Optional API ports =====
    DOC_API_PORT: Optional[int] = None
    RAG_API_PORT: Optional[int] = None

    # Config
    if V2:
        model_config = {"env_file": ".env", "extra": "ignore"}
    else:
        class Config:
            env_file = ".env"
            extra = "ignore"

    def get_public_url(self, internal_url: str) -> str:
        """
        Convert internal MinIO URL to public ngrok URL

        Args:
            internal_url: http://localhost:9000/public-documents/file.pdf

        Returns:
            https://b1234567.ngrok-free.app/public-documents/file.pdf
        """
        if not self.ENABLE_URL_REPLACEMENT:
            return internal_url

        if not self.NGROK_PUBLIC_URL:
            return internal_url

        # Replace internal URL with ngrok URL
        if internal_url.startswith(self.MINIO_INTERNAL_URL):
            return internal_url.replace(
                self.MINIO_INTERNAL_URL,
                self.NGROK_PUBLIC_URL.rstrip('/')
            )

        return internal_url


# Singleton settings
settings = Settings()


# Helper function to get FAQ config
def get_faq_config() -> dict:
    """Get FAQ-specific configuration"""
    return {
        "vector_threshold": settings.FAQ_VECTOR_THRESHOLD,
        "rerank_threshold": settings.FAQ_RERANK_THRESHOLD,
        "top_k": settings.FAQ_TOP_K,
        "weights": {
            "question": settings.FAQ_QUESTION_WEIGHT,
            "question_answer": settings.FAQ_QA_WEIGHT,
            "answer": settings.FAQ_ANSWER_WEIGHT
        },
        "consistency_bonus": settings.FAQ_CONSISTENCY_BONUS,
        "consistency_threshold": settings.FAQ_CONSISTENCY_THRESHOLD
    }
