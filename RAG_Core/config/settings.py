# RAG_Core/config/settings.py

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
    DOCUMENT_URLS_COLLECTION: str = "document_urls"

    DOCUMENT_VECTOR_FIELD: str = "content_vector"
    FAQ_VECTOR_FIELD: str = "question_vector"

    # ===== Ollama / LLM =====
    OLLAMA_URL: str = "http://ollama:11434"
    LLM_MODEL: str = "gpt-oss:20b"
    OLLAMA_BASE_URL: Optional[str] = None

    # ===== Embedding =====
    EMBEDDING_MODEL: str = "keepitreal/vietnamese-sbert"
    EMBEDDING_DIM: int = 768

    # ===== Reranker (local, không call API) =====
    # BAAI/bge-reranker-v2-m3: multilingual cross-encoder, tốt nhất cho tiếng Việt
    # Thay thế hoàn toàn Cohere Rerank API
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    RERANKER_USE_FP16: bool = True
    RERANKER_MAX_LENGTH: int = 1024
    RERANKER_DEVICE: str = "cuda:0"   # ← ép 1 GPU, tắt multi-process pool
    RERANKER_BATCH_SIZE: int = 32     # ← batch size cho single-process inference

    # ===== Search / RAG =====
    SIMILARITY_THRESHOLD: float = 0.2
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

    # ===== DOCUMENT URL SETTINGS =====
    MINIO_INTERNAL_URL: str = "http://localhost:9000"
    NGROK_PUBLIC_URL: Optional[str] = "http://124.158.6.101:9000"
    ENABLE_URL_REPLACEMENT: bool = True
    URL_FORMAT_STYLE: str = "detailed"
    MAX_REFERENCE_URLS: int = 5

    # ===== Contact =====
    SUPPORT_PHONE: str = "00-84-24-36417184"

    # ===== Identity =====
    BANK_FULL_NAME: str = "Ngân hàng Chính sách Xã hội Việt Nam (VBSP)"
    ASSISTANT_NAME: str = "Trợ lý ảo nội bộ VBSP"

    # ===== API port =====
    RAG_API_PORT: int = 8522
    DOC_API_PORT: Optional[int] = None

    if V2:
        model_config = {"env_file": ".env", "extra": "ignore"}
    else:
        class Config:
            env_file = ".env"
            extra = "ignore"

    def get_public_url(self, internal_url: str) -> str:
        if not self.ENABLE_URL_REPLACEMENT:
            return internal_url
        if not self.NGROK_PUBLIC_URL:
            return internal_url
        if internal_url.startswith(self.MINIO_INTERNAL_URL):
            return internal_url.replace(
                self.MINIO_INTERNAL_URL,
                self.NGROK_PUBLIC_URL.rstrip('/')
            )
        return internal_url


settings = Settings()


def get_faq_config() -> dict:
    return {
        "vector_threshold": settings.FAQ_VECTOR_THRESHOLD,
        "rerank_threshold": settings.FAQ_RERANK_THRESHOLD,
        "top_k": settings.FAQ_TOP_K,
        "weights": {
            "question": settings.FAQ_QUESTION_WEIGHT,
            "question_answer": settings.FAQ_QA_WEIGHT,
            "answer": settings.FAQ_ANSWER_WEIGHT,
        },
        "consistency_bonus": settings.FAQ_CONSISTENCY_BONUS,
        "consistency_threshold": settings.FAQ_CONSISTENCY_THRESHOLD,
    }