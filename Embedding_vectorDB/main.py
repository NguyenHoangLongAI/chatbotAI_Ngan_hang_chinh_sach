#!/usr/bin/env python3
"""
main.py — Unified Document Processing API v6
=============================================
Thay đổi so với v5:
  - Engine: PaddleOCR v3.7.0 (PP-OCRv5 + PP-StructureV3) thay vì PaddleOCR-VL
  - ProtonX Vietnamese text correction tích hợp trong processor
  - Thêm /api/v1/correct-text endpoint để test correction standalone
  - Thêm correction_model param trong /api/v1/process-document
  - Cấu hình qua config.py (CORRECTION_MODEL, OCR_LANG, v.v.)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
import uuid
import re
import io
import logging
from typing import Optional, List, Dict, Any

from document_processor import DocumentProcessor, ChunkRecord
from embedding_service import EmbeddingService
from milvus_client import MilvusManager
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified Document Processing API",
    version="6.0.0",
    description=(
        "PaddleOCR v3.7.0 (PP-OCRv5 + PP-StructureV3) + "
        "ProtonX Vietnamese Text Correction + "
        "Vietnamese SBERT 768D Embedding + Milvus + MinIO"
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# GLOBAL SERVICES
# =====================================================================

MILVUS_HOST = os.getenv("MILVUS_HOST", "10.22.14.6")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19532")

MINIO_INTERNAL_ENDPOINT = os.getenv("MINIO_INTERNAL_ENDPOINT", "10.22.14.6:19100")
MINIO_PUBLIC_ENDPOINT   = os.getenv("MINIO_PUBLIC_ENDPOINT",   "10.22.14.6:19100")
MINIO_ACCESS_KEY        = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY        = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET            = os.getenv("MINIO_BUCKET", "public-documents")
MINIO_SECURE            = os.getenv("MINIO_SECURE", "false").lower() == "true"

milvus_manager    = MilvusManager(host=MILVUS_HOST, port=MILVUS_PORT)
doc_processor     = DocumentProcessor(
    use_gpu=config.USE_GPU,
    correction_model=config.CORRECTION_MODEL,
    lang=config.OCR_LANG,
)
embedding_service = EmbeddingService()

_minio_client = None
_url_manager  = None


# =====================================================================
# HELPERS
# =====================================================================

def sanitize_id(text: str) -> str:
    s = re.sub(r"[^\w\-_.]", "_", text)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def get_safe_temp_filename(original_filename: str) -> str:
    _, ext = os.path.splitext(original_filename)
    return f"tmp_{uuid.uuid4().hex[:8]}{ext.lower()}"


def _chunk_preview(record: ChunkRecord, idx: int) -> Dict[str, Any]:
    flags = []
    if record.has_image:   flags.append("has_image")
    if record.has_table:   flags.append("has_table")
    if record.has_heading: flags.append("has_heading")
    if record.is_overlap:  flags.append("is_overlap")

    preview = record.content[:200].replace('\n', ' ')
    if len(record.content) > 200:
        preview += "..."

    return {
        "chunk_index":     idx,
        "milvus_id":       record.milvus_id,
        "chunk_type":      record.chunk_type,
        "section_path":    record.section_path,
        "page_num":        record.page_num,
        "part_index":      record.part_index,
        "token_count":     record.token_count,
        "char_count":      record.char_count,
        "flags":           flags,
        "content_preview": preview,
    }


# =====================================================================
# MINIO HELPERS (unchanged from v5)
# =====================================================================

def _get_minio_client():
    global _minio_client
    if _minio_client is not None:
        return _minio_client
    try:
        from minio import Minio
        import json

        client = Minio(
            MINIO_INTERNAL_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)

        policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"AWS": "*"},
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{MINIO_BUCKET}/*"],
            }],
        }
        client.set_bucket_policy(MINIO_BUCKET, json.dumps(policy))
        _minio_client = client
        logger.info(f"✅ MinIO ready: {MINIO_INTERNAL_ENDPOINT}/{MINIO_BUCKET}")
        return client
    except Exception as e:
        logger.warning(f"⚠️ MinIO init failed: {e}")
        return None


def _upload_to_minio(file_bytes: bytes, document_id: str, file_ext: str) -> Optional[str]:
    client = _get_minio_client()
    if client is None:
        return None

    CONTENT_TYPES = {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc":  "application/msword",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xls":  "application/vnd.ms-excel",
        ".txt":  "text/plain",
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".tiff": "image/tiff",
    }
    content_type = CONTENT_TYPES.get(file_ext, "application/octet-stream")
    object_name = f"{document_id}{file_ext}"

    try:
        client.put_object(
            MINIO_BUCKET, object_name,
            io.BytesIO(file_bytes), length=len(file_bytes),
            content_type=content_type,
        )
        protocol = "https" if MINIO_SECURE else "http"
        return f"{protocol}://{MINIO_PUBLIC_ENDPOINT}/{MINIO_BUCKET}/{object_name}"
    except Exception as e:
        logger.error(f"❌ MinIO upload error: {e}")
        return None


def _store_url_in_milvus(document_id: str, url: str, filename: str, file_ext: str) -> bool:
    try:
        from document_urls_collection import DocumentURLsManager
        global _url_manager
        if _url_manager is None:
            _url_manager = DocumentURLsManager(host=MILVUS_HOST, port=MILVUS_PORT)
            _url_manager.create_collection()
            _url_manager._embedding_model = embedding_service.model
        return _url_manager.insert_url(
            document_id=document_id, url=url,
            filename=filename, file_type=file_ext,
        )
    except Exception as e:
        logger.error(f"❌ URL store error: {e}")
        return False


# =====================================================================
# STARTUP
# =====================================================================

@app.on_event("startup")
async def startup_event():
    try:
        await milvus_manager.initialize()
        _get_minio_client()
        logger.info("✅ Document API v6 started")
        logger.info(f"   Milvus     : {MILVUS_HOST}:{MILVUS_PORT}")
        logger.info(f"   MinIO      : {MINIO_INTERNAL_ENDPOINT}/{MINIO_BUCKET}")
        logger.info(f"   OCR Engine : PaddleOCR v3.7.0 (PP-OCRv5 + PP-StructureV3)")
        logger.info(f"   Correction : ProtonX {config.CORRECTION_MODEL} model")
        logger.info(f"   OCR Lang   : {config.OCR_LANG}")
    except Exception as e:
        logger.error(f"⚠️ Startup warning: {e}")


# =====================================================================
# ENDPOINTS
# =====================================================================

@app.get("/")
async def root():
    return {
        "service": "Unified Document Processing API",
        "version": "6.0.0",
        "status": "running",
        "features": {
            "ocr_engine":        "PaddleOCR v3.7.0 (PP-OCRv5 + PP-StructureV3)",
            "text_correction":   f"ProtonX Vietnamese ({config.CORRECTION_MODEL} model)",
            "table_parsing":     "PP-StructureV3 HTML→Markdown",
            "smart_chunking":    "semantic v4 (heading/table/image/toc aware)",
            "embedding":         "Vietnamese SBERT 768D",
            "storage":           "Milvus (document_chunks) + MinIO + Milvus (document_urls)",
        },
        "supported_formats": [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".txt",
                               ".png", ".jpg", ".jpeg", ".tiff"],
        "collections": ["document_chunks", "faq_embeddings", "document_urls"],
    }


# ─────────────────────────────────────────────────────────────────────
# MAIN: Process document
# ─────────────────────────────────────────────────────────────────────

@app.post("/api/v1/process-document")
async def process_document(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
    chunk_mode: str = Form("smart"),
    upload_to_minio: bool = Form(True),
    correction_model: Optional[str] = Form(None),  # NEW: override model per request
):
    """
    Pipeline đầy đủ v6:
      Step 1: Lưu file tạm
      Step 2: Upload lên MinIO + store URL (song song với Step 3)
      Step 3: Extract text — PaddleOCR v3.7.0 (PP-OCRv5 + PP-StructureV3)
              + ProtonX Vietnamese text correction
      Step 4: SmartChunker v4 → ChunkRecord list
      Step 5: Embed từng chunk (Vietnamese SBERT 768D)
      Step 6: Insert vào document_chunks collection
    """
    temp_file_path = None

    try:
        if not file.filename:
            raise HTTPException(400, "No file provided")

        allowed_ext = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt",
                       ".png", ".jpg", ".jpeg", ".tiff"}
        original_filename = file.filename
        file_ext = os.path.splitext(original_filename)[1].lower()

        if file_ext not in allowed_ext:
            raise HTTPException(400, f"File type {file_ext} not supported")

        valid_modes = ["smart", "sentence", "legacy"]
        if chunk_mode not in valid_modes:
            raise HTTPException(400, f"chunk_mode must be one of: {valid_modes}")

        if document_id:
            document_id = sanitize_id(document_id)
        else:
            document_id = sanitize_id(os.path.splitext(original_filename)[0])
        if not document_id:
            document_id = f"doc_{uuid.uuid4().hex[:8]}"

        logger.info(f"📄 [{document_id}] Processing: {original_filename}")

        # ── Step 1: Save temp ──────────────────────────────────────
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(400, "File is empty")

        safe_name = get_safe_temp_filename(original_filename)
        temp_file_path = os.path.join(tempfile.gettempdir(), safe_name)
        with open(temp_file_path, "wb") as f:
            f.write(content)
        logger.info(f"✅ [1/5] Saved temp ({len(content)} bytes)")

        # ── Step 2: Upload to MinIO ────────────────────────────────
        public_url = None
        url_stored = False
        if upload_to_minio:
            public_url = _upload_to_minio(content, document_id, file_ext)
            if public_url:
                url_stored = _store_url_in_milvus(
                    document_id, public_url, original_filename, file_ext
                )
                logger.info(f"✅ [2/5] MinIO URL stored: {url_stored}")
        else:
            logger.info("⏭️ [2/5] Skipping MinIO upload")

        # ── Step 3: Extract text ───────────────────────────────────
        logger.info(f"🔍 [3/5] Extracting text ({file_ext})...")

        # Override correction model nếu được chỉ định
        active_processor = doc_processor
        if correction_model and correction_model != config.CORRECTION_MODEL:
            valid_models = ["teacher", "student", "nano"]
            if correction_model not in valid_models:
                raise HTTPException(400, f"correction_model must be one of: {valid_models}")
            # Re-init processor với model mới (có thể cache sau)
            active_processor = DocumentProcessor(
                use_gpu=config.USE_GPU,
                correction_model=correction_model,
                lang=config.OCR_LANG,
            )

        if file_ext == ".pdf":
            markdown_content = active_processor.process_pdf(temp_file_path)
        elif file_ext in (".doc", ".docx"):
            markdown_content = active_processor.process_word(temp_file_path)
        elif file_ext in (".xls", ".xlsx"):
            markdown_content = active_processor.process_excel(temp_file_path)
        elif file_ext in (".png", ".jpg", ".jpeg", ".tiff"):
            markdown_content = active_processor.process_image(temp_file_path)
        elif file_ext == ".txt":
            text_content = None
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    with open(temp_file_path, "r", encoding=enc) as f:
                        text_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            if text_content is None:
                raise HTTPException(400, "Cannot decode text file")
            markdown_content = active_processor.process_text(text_content)
        else:
            raise HTTPException(422, f"Unsupported extension: {file_ext}")

        if not markdown_content or not markdown_content.strip():
            raise HTTPException(422, "Could not extract content from file")

        logger.info(f"✅ [3/5] Extracted {len(markdown_content)} chars")

        # ── Step 4: Smart Chunking ────────────────────────────────
        logger.info(f"🧩 [4/5] Chunking (mode={chunk_mode})...")
        chunk_records: List[ChunkRecord] = active_processor.parse_markdown_to_chunk_records(
            markdown_content, document_id, chunk_mode
        )
        if not chunk_records:
            raise HTTPException(422, "Could not parse markdown into chunks")
        logger.info(f"✅ [4/5] Created {len(chunk_records)} chunks")

        # ── Step 5: Embed ─────────────────────────────────────────────
        logger.info(f"🔗 [5/5] Embedding {len(chunk_records)} chunks...")
        texts = [r.content_with_ctx for r in chunk_records]
        try:
            # Batch embed tất cả cùng lúc — nhanh hơn nhiều
            import torch
            with torch.no_grad():
                vectors = embedding_service.model.encode(
                    texts,
                    batch_size=32,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            for record, vec in zip(chunk_records, vectors):
                record.content_vector = vec.tolist()
            embed_ok, embed_fail = len(chunk_records), 0
        except Exception as e:
            logger.warning(f"Batch embed failed, falling back to per-chunk: {e}")
            embed_ok = embed_fail = 0
            for record in chunk_records:
                try:
                    record.content_vector = embedding_service.get_embedding(record.content_with_ctx)
                    embed_ok += 1
                except Exception as e2:
                    record.content_vector = [0.0] * 768
                    embed_fail += 1

        # ── Step 6: Insert to Milvus ──────────────────────────────
        stored_count = await milvus_manager.insert_chunks(chunk_records)
        logger.info(f"✅ [5/5] Stored {stored_count}/{len(chunk_records)} vectors")

        # ── Build response ────────────────────────────────────────
        chunk_previews = [_chunk_preview(r, r.chunk_index) for r in chunk_records]

        from collections import Counter
        type_summary = dict(Counter(r.chunk_type for r in chunk_records))
        flag_summary = {
            "has_image":   sum(r.has_image   for r in chunk_records),
            "has_table":   sum(r.has_table   for r in chunk_records),
            "has_heading": sum(r.has_heading for r in chunk_records),
            "is_overlap":  sum(r.is_overlap  for r in chunk_records),
        }
        pages_detected = len(set(r.page_num for r in chunk_records if r.page_num > 0))

        return {
            "status": "success",
            "message": "Document processed successfully",
            "document_info": {
                "document_id":       document_id,
                "original_filename": original_filename,
                "file_type":         file_ext,
                "file_size_bytes":   len(content),
                "public_url":        public_url,
                "url_stored":        url_stored,
            },
            "processing_stats": {
                "ocr_engine":          "PaddleOCR v3.7.0 (PP-OCRv5 + PP-StructureV3)",
                "correction_model":    correction_model or config.CORRECTION_MODEL,
                "markdown_length":     len(markdown_content),
                "pages_detected":      pages_detected,
                "total_chunks":        len(chunk_records),
                "embed_success":       embed_ok,
                "embed_failed":        embed_fail,
                "stored_in_milvus":    stored_count,
                "chunk_mode":          chunk_mode,
                "chunk_type_summary":  type_summary,
                "flag_summary":        flag_summary,
            },
            "chunk_previews": chunk_previews,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        raise HTTPException(500, f"Processing error: {str(e)}")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────
# NEW: Vietnamese Text Correction endpoint (test/debug)
# ─────────────────────────────────────────────────────────────────────

@app.post("/api/v1/correct-text")
async def correct_vietnamese_text(request: dict):
    """
    Sửa chính tả tiếng Việt bằng ProtonX model.
    Dùng để test correction standalone hoặc correct text đã có.

    Request body:
      {
        "text": "van ban can sua chinh ta",
        "model": "student"   // optional: teacher / student / nano
      }
    """
    try:
        text = request.get("text", "").strip()
        model_size = request.get("model", config.CORRECTION_MODEL)

        if not text:
            raise HTTPException(400, "text is required")

        valid_models = ["teacher", "student", "nano"]
        if model_size not in valid_models:
            raise HTTPException(400, f"model must be one of: {valid_models}")

        from paddle_ocr_processor import VietnameseTextCorrector
        corrector = VietnameseTextCorrector(model_size=model_size, enabled=True)
        corrected = corrector.correct(text)

        return {
            "status": "success",
            "original": text,
            "corrected": corrected,
            "model": model_size,
            "changed": text != corrected,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Correction error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────
# FAQ endpoints (unchanged)
# ─────────────────────────────────────────────────────────────────────

@app.post("/api/v1/faq/add")
async def add_faq(request: dict):
    try:
        question = request.get("question", "").strip()
        answer   = request.get("answer", "").strip()
        faq_id   = request.get("faq_id", "").strip()

        if not question: raise HTTPException(400, "Question is required")
        if not answer:   raise HTTPException(400, "Answer is required")
        if not faq_id:
            faq_id = f"faq_{uuid.uuid4().hex[:8]}"
        else:
            faq_id = sanitize_id(faq_id)

        question_embedding = embedding_service.get_embedding(question)
        await milvus_manager.insert_faq(faq_id, question, answer, question_embedding)
        return {"status": "success", "faq_id": faq_id, "question": question}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Add FAQ error: {str(e)}")


@app.delete("/api/v1/faq/delete/{faq_id}")
async def delete_faq(faq_id: str):
    try:
        await milvus_manager.delete_faq(faq_id.strip())
        return {"status": "success", "faq_id": faq_id}
    except Exception as e:
        raise HTTPException(500, f"Delete FAQ error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────
# Document management (unchanged)
# ─────────────────────────────────────────────────────────────────────

@app.delete("/api/v1/document/delete/{document_id}")
async def delete_document(document_id: str):
    try:
        await milvus_manager.delete_document(document_id.strip())
        return {"status": "success", "document_id": document_id}
    except Exception as e:
        raise HTTPException(500, f"Delete error: {str(e)}")


@app.get("/api/v1/document/url/{document_id}")
async def get_document_url(document_id: str):
    try:
        from document_urls_collection import DocumentURLsManager
        global _url_manager
        if _url_manager is None:
            _url_manager = DocumentURLsManager(host=MILVUS_HOST, port=MILVUS_PORT)
            _url_manager.create_collection()
        info = _url_manager.get_url(document_id.strip())
        if not info:
            raise HTTPException(404, f"No URL found for document_id: {document_id}")
        return {"status": "success", **info}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Get URL error: {str(e)}")


@app.post("/api/v1/document/upload-url")
async def upload_url_only(
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(None),
):
    try:
        content = await file.read()
        if not content:
            raise HTTPException(400, "File is empty")

        original_filename = file.filename or "unknown"
        file_ext = os.path.splitext(original_filename)[1].lower()

        if document_id:
            doc_id = sanitize_id(document_id)
        else:
            doc_id = sanitize_id(os.path.splitext(original_filename)[0])
        if not doc_id:
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"

        public_url = _upload_to_minio(content, doc_id, file_ext)
        if not public_url:
            raise HTTPException(500, "MinIO upload failed")

        url_stored = _store_url_in_milvus(doc_id, public_url, original_filename, file_ext)

        return {
            "status": "success",
            "document_id": doc_id,
            "original_filename": original_filename,
            "file_type": file_ext,
            "file_size_bytes": len(content),
            "public_url": public_url,
            "url_stored_in_milvus": url_stored,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload error: {str(e)}")


# ─────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health_check():
    try:
        milvus_ok  = await milvus_manager.health_check()
        embed_ok   = embedding_service.is_ready()
        paddle_ok  = doc_processor.paddle_processor is not None
        minio_ok   = _get_minio_client() is not None
        correct_ok = (
            doc_processor.paddle_processor._corrector.enabled
            if paddle_ok else False
        )

        all_ok = all([milvus_ok, embed_ok, paddle_ok])
        return {
            "status": "healthy" if all_ok else "degraded",
            "service": "unified-document-api",
            "version": "6.0.0",
            "services": {
                "milvus":              milvus_ok,
                "embedding_model":     embed_ok,
                "paddle_ocr_v3":       paddle_ok,
                "pp_structure_v3":     paddle_ok,
                "protonx_correction":  correct_ok,
                "minio":               minio_ok,
            },
            "config": {
                "ocr_engine":       "PaddleOCR v3.7.0 (PP-OCRv5)",
                "correction_model": config.CORRECTION_MODEL,
                "ocr_lang":         config.OCR_LANG,
                "use_gpu":          config.USE_GPU,
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8022, log_level="info")