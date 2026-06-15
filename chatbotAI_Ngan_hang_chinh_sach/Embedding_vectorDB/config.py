# Embedding_vectorDB/config.py
import os


class Config:
    # ===== Milvus =====
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "document_embeddings")
    FAQ_COLLECTION: str = os.getenv("FAQ_COLLECTION", "faq_embeddings")

    # ===== Embedding =====
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
    EMBEDDING_DIMENSION: int = int(
        os.getenv("EMBEDDING_DIM", os.getenv("EMBEDDING_DIMENSION", "768"))
    )

    # ===== Document Processing =====
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "100")) * 1024 * 1024  # 100MB
    TESSDATA_PREFIX: str = os.getenv(
        "TESSDATA_PREFIX", "/usr/share/tesseract-ocr/4.00/tessdata"
    )

    # ===== PADDLEOCR-VL SETTINGS =====
    # Enable/disable PaddleOCR-VL processor (replaces Docling)
    USE_PADDLE_OCR: bool = os.getenv("USE_PADDLE_OCR", "true").lower() == "true"

    # PaddleOCR-VL model ID (HuggingFace)
    PADDLE_OCR_MODEL_ID: str = os.getenv(
        "PADDLE_OCR_MODEL_ID", "PaddlePaddle/PaddleOCR-VL"
    )

    # Use GPU for VL model inference
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() == "true"

    # Image scale for PDF rendering (higher = better quality, slower)
    # 1.0 = 108 DPI, 2.0 = 216 DPI, 3.0 = 324 DPI
    IMAGE_SCALE: float = float(os.getenv("IMAGE_SCALE", "2.0"))

    # Max new tokens for VL model generation per page
    VL_MAX_NEW_TOKENS: int = int(os.getenv("VL_MAX_NEW_TOKENS", "4096"))

    # Enable rule-based spell correction post-processing
    ENABLE_SPELL_CORRECTION: bool = os.getenv("ENABLE_SPELL_CORRECTION", "true").lower() == "true"

    # Enable table normalization post-processing
    ENABLE_TABLE_NORMALIZATION: bool = os.getenv("ENABLE_TABLE_NORMALIZATION", "true").lower() == "true"

    # Legacy OCR fallback (Tesseract) - used when PaddleOCR-VL fails
    USE_LEGACY_OCR_FALLBACK: bool = os.getenv("USE_LEGACY_OCR_FALLBACK", "true").lower() == "true"
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "vi,en")

    @property
    def ocr_lang_list(self) -> list:
        return [lang.strip() for lang in self.OCR_LANGUAGES.split(',')]


config = Config()