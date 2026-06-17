# Embedding_vectorDB/config.py
import os


class Config:
    # ===== Milvus =====
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "document_chunks")
    FAQ_COLLECTION: str = os.getenv("FAQ_COLLECTION", "faq_embeddings")

    # ===== Embedding =====
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
    EMBEDDING_DIMENSION: int = int(
        os.getenv("EMBEDDING_DIM", os.getenv("EMBEDDING_DIMENSION", "768"))
    )

    # ===== Document Processing =====
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", "100")) * 1024 * 1024  # 100MB

    # ===== PADDLEOCR v3.7.0 SETTINGS =====
    # PaddleOCR version: 3.7.0 (PyPI) with PP-OCRv5 + PP-StructureV3
    USE_GPU: bool = os.getenv("USE_GPU", "true").lower() == "true"

    # PP-OCRv5 language (vi = Vietnamese, ch = Chinese+English, en = English)
    OCR_LANG: str = os.getenv("OCR_LANG", "vi")

    # Image scale for PDF rendering (1.5 = 225 DPI — good balance quality/speed)
    IMAGE_SCALE: float = float(os.getenv("IMAGE_SCALE", "1.5"))

    # Enable PP-StructureV3 for table + layout analysis (recommended: true)
    ENABLE_PP_STRUCTURE: bool = os.getenv("ENABLE_PP_STRUCTURE", "true").lower() == "true"

    # ===== PROTONX VIETNAMESE TEXT CORRECTION =====
    # Enable ProtonX text correction post-processing
    ENABLE_SPELL_CORRECTION: bool = os.getenv("ENABLE_SPELL_CORRECTION", "true").lower() == "true"

    # ProtonX correction model size:
    #   "teacher"  → protonx-models/protonx-legal-tc         (904MB, ROUGE-L 98.44)
    #   "student"  → protonx-models/distilled-protonx-legal-tc (507MB, ROUGE-L 97.64)
    #   "nano"     → protonx-models/nano-protonx-legal-tc    (smallest, fastest)
    CORRECTION_MODEL: str = os.getenv("CORRECTION_MODEL", "student")

    # ===== TABLE PROCESSING =====
    # Enable table normalization (TSV / aligned-space → Markdown)
    ENABLE_TABLE_NORMALIZATION: bool = os.getenv("ENABLE_TABLE_NORMALIZATION", "true").lower() == "true"

    # ===== LEGACY FALLBACK =====
    # Tesseract fallback nếu PaddleOCR fail
    USE_LEGACY_OCR_FALLBACK: bool = os.getenv("USE_LEGACY_OCR_FALLBACK", "false").lower() == "true"
    TESSDATA_PREFIX: str = os.getenv(
        "TESSDATA_PREFIX", "/usr/share/tesseract-ocr/4.00/tessdata"
    )
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "vi,en")

    @property
    def ocr_lang_list(self) -> list:
        return [lang.strip() for lang in self.OCR_LANGUAGES.split(',')]


config = Config()