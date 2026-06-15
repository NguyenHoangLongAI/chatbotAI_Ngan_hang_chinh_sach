#!/usr/bin/env python3
"""
main.py — Unified Document Processing API v5
=============================================
Thay đổi chính so với v4:
  - Collection: document_chunks (thay vì document_embeddings)
  - ChunkRecord thay Dict — metadata đầy đủ
  - MinIO URL lưu vào document_urls collection ngay sau upload
  - Response của /api/v1/process-document trả về chunk_previews
    (description từng chunk: type, section, content preview, flags)
  - MilvusManager.insert_chunks() thay insert_embeddings()
  - /api/v1/document/url/{document_id} endpoint mới
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

app = FastAPI(
    title="Unified Document Processing API",
    version="5.0.0",
    description="Process + Embed + Store in Milvus + MinIO URL storage",
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

milvus_manager  = MilvusManager(host=MILVUS_HOST, port=MILVUS_PORT)
doc_processor   = DocumentProcessor(use_gpu=os.getenv("USE_GPU", "true").lower() == "true")
embedding_service = EmbeddingService()

# MinIO client (initialized at startup)
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
    """Tạo description preview cho một chunk để trả về trong response."""
    flags = []
    if record.has_image:   flags.append("has_image")
    if record.has_table:   flags.append("has_table")
    if record.has_heading: flags.append("has_heading")
    if record.is_overlap:  flags.append("is_overlap")

    # Preview: 200 chars của content (không kể header)
    preview = record.content[:200].replace('\n', ' ')
    if len(record.content) > 200:
        preview += "..."

    return {
        "chunk_index":    idx,
        "milvus_id":      record.milvus_id,
        "chunk_type":     record.chunk_type,
        "section_path":   record.section_path,
        "page_num":       record.page_num,
        "part_index":     record.part_index,
        "token_count":    record.token_count,
        "char_count":     record.char_count,
        "flags":          flags,
        "content_preview": preview,
    }


# =====================================================================
# MINIO HELPERS
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
        # Ensure bucket exists
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
            logger.info(f"✅ MinIO bucket created: {MINIO_BUCKET}")

        # Set public read policy
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
    """Upload bytes lên MinIO, trả về public URL hoặc None."""
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
            MINIO_BUCKET,
            object_name,
            io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=content_type,
        )
        protocol = "https" if MINIO_SECURE else "http"
        public_url = f"{protocol}://{MINIO_PUBLIC_ENDPOINT}/{MINIO_BUCKET}/{object_name}"
        logger.info(f"✅ MinIO uploaded: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"❌ MinIO upload error: {e}")
        return None


def _store_url_in_milvus(document_id: str, url: str, filename: str, file_ext: str) -> bool:
    """Lưu URL vào document_urls collection thông qua DocumentURLsManager."""
    try:
        from document_urls_collection import DocumentURLsManager
        global _url_manager
        if _url_manager is None:
            _url_manager = DocumentURLsManager(host=MILVUS_HOST, port=MILVUS_PORT)
            _url_manager.create_collection()

        return _url_manager.insert_url(
            document_id=document_id,
            url=url,
            filename=filename,
            file_type=file_ext,
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
        logger.info("✅ Document API v5 started")
        logger.info(f"   Milvus : {MILVUS_HOST}:{MILVUS_PORT}")
        logger.info(f"   MinIO  : {MINIO_INTERNAL_ENDPOINT}/{MINIO_BUCKET}")
    except Exception as e:
        logger.error(f"⚠️ Startup warning: {e}")


# =====================================================================
# ENDPOINTS
# =====================================================================

@app.get("/")
async def root():
    return {
        "service": "Unified Document Processing API",
        "version": "5.0.0",
        "status": "running",
        "features": {
            "document_processing": "PaddleOCR-VL Hybrid v3",
            "smart_chunking": "semantic v2 (heading/table/image aware)",
            "embedding": "Vietnamese SBERT 768D",
            "storage": "Milvus (document_chunks) + MinIO (files) + Milvus (document_urls)",
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
):
    """
    Pipeline đầy đủ:
      Step 1: Lưu file tạm
      Step 2: Upload lên MinIO + store URL vào Milvus (song song với Step 3)
      Step 3: Extract text via PaddleOCR-VL
      Step 4: SmartChunker v2 → ChunkRecord list
      Step 5: Embed từng chunk
      Step 6: Insert vào document_chunks collection
      Response: chunk_previews — description từng chunk
    """
    temp_file_path = None

    try:
        # ── Validate ──────────────────────────────────────────────────
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

        # ── Step 1: Save temp ─────────────────────────────────────────
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(400, "File is empty")

        safe_name = get_safe_temp_filename(original_filename)
        temp_file_path = os.path.join(tempfile.gettempdir(), safe_name)
        with open(temp_file_path, "wb") as f:
            f.write(content)
        logger.info(f"✅ [1/5] Saved temp ({len(content)} bytes)")

        # ── Step 2: Upload to MinIO + store URL ──────────────────────
        public_url = None
        url_stored = False
        if upload_to_minio:
            logger.info(f"📤 [2/5] Uploading to MinIO...")
            public_url = _upload_to_minio(content, document_id, file_ext)
            if public_url:
                url_stored = _store_url_in_milvus(
                    document_id, public_url, original_filename, file_ext
                )
                logger.info(f"✅ [2/5] MinIO URL stored: {url_stored}")
            else:
                logger.warning("⚠️ [2/5] MinIO upload failed — continuing without URL")
        else:
            logger.info("⏭️ [2/5] Skipping MinIO upload (upload_to_minio=False)")

        # ── Step 3: Extract text ──────────────────────────────────────
        logger.info(f"🔍 [3/5] Extracting text ({file_ext})...")
        if file_ext == ".pdf":
            markdown_content = doc_processor.process_pdf(temp_file_path)
        elif file_ext in (".doc", ".docx"):
            markdown_content = doc_processor.process_word(temp_file_path)
        elif file_ext in (".xls", ".xlsx"):
            markdown_content = doc_processor.process_excel(temp_file_path)
        elif file_ext in (".png", ".jpg", ".jpeg", ".tiff"):
            markdown_content = doc_processor.process_image(temp_file_path)
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
            markdown_content = doc_processor.process_text(text_content)
        else:
            raise HTTPException(422, f"Unsupported extension: {file_ext}")

        if not markdown_content or not markdown_content.strip():
            raise HTTPException(422, "Could not extract content from file")

        logger.info(f"✅ [3/5] Extracted {len(markdown_content)} chars")

        # ── Step 4: Smart Chunking v2 ─────────────────────────────────
        logger.info(f"🧩 [4/5] Chunking (mode={chunk_mode})...")
        chunk_records: List[ChunkRecord] = doc_processor.parse_markdown_to_chunk_records(
            markdown_content, document_id, chunk_mode
        )
        if not chunk_records:
            raise HTTPException(422, "Could not parse markdown into chunks")
        logger.info(f"✅ [4/5] Created {len(chunk_records)} chunks")

        # ── Step 5: Embed ─────────────────────────────────────────────
        logger.info(f"🔗 [5/5] Embedding {len(chunk_records)} chunks...")
        embed_ok = 0
        embed_fail = 0
        for record in chunk_records:
            try:
                # Embed content_with_ctx (includes context header)
                record.content_vector = embedding_service.get_embedding(
                    record.content_with_ctx
                )
                embed_ok += 1
            except Exception as e:
                logger.warning(f"  Embed failed chunk {record.chunk_index}: {e}")
                record.content_vector = [0.0] * 768
                embed_fail += 1

        # ── Step 6: Insert to Milvus ──────────────────────────────────
        stored_count = await milvus_manager.insert_chunks(chunk_records)
        logger.info(f"✅ [5/5] Stored {stored_count}/{len(chunk_records)} vectors")

        # ── Build response ────────────────────────────────────────────
        chunk_previews = [_chunk_preview(r, r.chunk_index) for r in chunk_records]

        # Group summary by type
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
                "document_id":      document_id,
                "original_filename": original_filename,
                "file_type":        file_ext,
                "file_size_bytes":  len(content),
                "public_url":       public_url,
                "url_stored":       url_stored,
            },
            "processing_stats": {
                "markdown_length":       len(markdown_content),
                "pages_detected":        pages_detected,
                "total_chunks":          len(chunk_records),
                "embed_success":         embed_ok,
                "embed_failed":          embed_fail,
                "stored_in_milvus":      stored_count,
                "chunk_mode":            chunk_mode,
                "chunk_type_summary":    type_summary,
                "flag_summary":          flag_summary,
            },
            "chunk_previews": chunk_previews,   # ← description từng chunk
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
        return {"status": "success", "faq_id": faq_id, "question": question, "answer": answer}
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
# Document management
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
    """Lấy public URL của một document từ document_urls collection."""
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
    """
    Chỉ upload file lên MinIO + lưu URL vào Milvus.
    KHÔNG process document / tạo embeddings.
    Dùng khi muốn tách bước upload khỏi bước embed.
    """
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
        milvus_ok   = await milvus_manager.health_check()
        embed_ok    = embedding_service.is_ready()
        paddle_ok   = doc_processor.paddle_processor is not None
        minio_ok    = _get_minio_client() is not None

        all_ok = all([milvus_ok, embed_ok, paddle_ok])
        return {
            "status": "healthy" if all_ok else "degraded",
            "service": "unified-document-api",
            "version": "5.0.0",
            "services": {
                "milvus":        milvus_ok,
                "embedding_model": embed_ok,
                "paddle_ocr_vl": paddle_ok,
                "minio":         minio_ok,
                "smart_chunking": "v2",
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8022, log_level="info")