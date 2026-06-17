#!/usr/bin/env python3
"""
create_collections.py — Vietnamese Document Pipeline (v2)
=========================================================
Collections:
  1. document_chunks   — chunks with FULL metadata (heading, table, base64, page, etc.)
  2. faq_embeddings    — FAQ questions (768D HNSW COSINE)
  3. document_urls     — URL + filename embedding (768D IVF_FLAT COSINE)

Schema thay đổi so với v1:
  - document_embeddings → document_chunks (thêm metadata: chunk_type, page_num,
    section_path, has_table, has_image, char_count, token_count, part_index)
  - Bỏ field 'description' dài 65000 → tách thành 'content' (text thuần)
    + 'content_with_context' (có header prefix)
  - Thêm các flag boolean dạng INT8 (Milvus chưa có BOOL native)

Usage:
    python create_collections.py
    python create_collections.py --host 10.22.14.6 --port 19532
    python create_collections.py --drop-existing
    python create_collections.py --collection document_chunks
"""

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class CollectionConfig:
    name: str
    description: str
    embedding_dim: int = 768


COLLECTIONS = [
    CollectionConfig("document_chunks",  "Document chunks with full metadata — Vietnamese SBERT 768D"),
    CollectionConfig("faq_embeddings",   "FAQ question embeddings — Vietnamese SBERT 768D"),
    CollectionConfig("document_urls",    "Document URLs + filename embeddings — Vietnamese SBERT 768D"),
]


# ============================================================================
# MILVUS HELPERS
# ============================================================================

def connect(host: str, port: str) -> None:
    from pymilvus import connections
    try:
        connections.disconnect("default")
    except Exception:
        pass
    connections.connect("default", host=host, port=int(port))
    logger.info(f"✅ Connected to Milvus at {host}:{port}")


def drop_collection(name: str) -> None:
    from pymilvus import utility, Collection
    if utility.has_collection(name):
        Collection(name).drop()
        logger.info(f"🗑️  Dropped: {name}")


def collection_exists(name: str) -> bool:
    from pymilvus import utility
    return utility.has_collection(name)


# ============================================================================
# INDEX HELPERS
# ============================================================================

def hnsw_index(field: str) -> dict:
    return {
        "field_name": field,
        "index_params": {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200},
        },
    }


def ivf_index(field: str) -> dict:
    return {
        "field_name": field,
        "index_params": {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        },
    }


# ============================================================================
# COLLECTION 1: document_chunks  (MAIN — replaces document_embeddings)
# ============================================================================

def create_document_chunks(dim: int = 768):
    """
    document_chunks — schema đầy đủ metadata:

    PK / Identity
    ─────────────
    id                  VARCHAR(220)   "{document_id}__{chunk_mode}__{chunk_index}"
    document_id         VARCHAR(100)   tên file (không extension)

    Content
    ───────
    content             VARCHAR(8000)  text thuần của chunk (không có header prefix)
    content_with_ctx    VARCHAR(10000) content có context header (dùng để embed + hiển thị)

    Context / Structure
    ───────────────────
    section_path        VARCHAR(500)   "Chương 1 > Mục 1.2 > ..." (breadcrumb)
    context_header      VARCHAR(1000)  markdown headings dẫn đến section này
    chunk_type          VARCHAR(30)    complete_section | partial_section | sentence_group
                                       | table | image | heading_only
    page_num            INT32          trang PDF (0 = unknown)
    part_index          INT16          thứ tự chunk trong cùng section (0-based)
    total_parts         INT16          tổng số chunks của section này

    Metrics
    ───────
    token_count         INT32          ước tính số token
    char_count          INT32          số ký tự content (không tính header)

    Flags (0/1 — Milvus INT8)
    ─────────────────────────
    has_table           INT8           chunk chứa markdown table
    has_image           INT8           chunk chứa base64 image block
    has_heading         INT8           chunk bắt đầu bằng # heading
    is_overlap          INT8           chunk có overlap từ chunk trước

    Vector
    ──────
    content_vector      FLOAT_VECTOR(768)   embedding của content_with_ctx
    """
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

    name = "document_chunks"
    fields = [
        # --- Identity ---
        FieldSchema("id",               DataType.VARCHAR, max_length=220,  is_primary=True),
        FieldSchema("document_id",      DataType.VARCHAR, max_length=100),

        # --- Content ---
        FieldSchema("content",          DataType.VARCHAR, max_length=65000),
        FieldSchema("content_with_ctx", DataType.VARCHAR, max_length=65000),

        # --- Context / Structure ---
        FieldSchema("section_path",     DataType.VARCHAR, max_length=500),
        FieldSchema("context_header",   DataType.VARCHAR, max_length=1000),
        FieldSchema("chunk_type",       DataType.VARCHAR, max_length=30),
        FieldSchema("page_num",         DataType.INT32),
        FieldSchema("part_index",       DataType.INT16),
        FieldSchema("total_parts",      DataType.INT16),

        # --- Metrics ---
        FieldSchema("token_count",      DataType.INT32),
        FieldSchema("char_count",       DataType.INT32),

        # --- Flags ---
        FieldSchema("has_table",        DataType.INT8),
        FieldSchema("has_image",        DataType.INT8),
        FieldSchema("has_heading",      DataType.INT8),
        FieldSchema("is_overlap",       DataType.INT8),

        # --- Vector ---
        FieldSchema("content_vector",   DataType.FLOAT_VECTOR, dim=dim),
    ]

    schema = CollectionSchema(fields, description="Document chunks — full metadata 768D")
    col = Collection(name=name, schema=schema)
    col.create_index(**hnsw_index("content_vector"))
    col.load()

    logger.info("  Schema document_chunks:")
    logger.info("    Identity : id | document_id")
    logger.info("    Content  : content | content_with_ctx")
    logger.info("    Structure: section_path | context_header | chunk_type | page_num | part_index | total_parts")
    logger.info("    Metrics  : token_count | char_count")
    logger.info("    Flags    : has_table | has_image | has_heading | is_overlap")
    logger.info(f"    Vector   : content_vector({dim}D) HNSW M=16 efConstruction=200 COSINE")
    return col


# ============================================================================
# COLLECTION 2: faq_embeddings  (unchanged schema, same as before)
# ============================================================================

def create_faq_embeddings(dim: int = 768):
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

    name = "faq_embeddings"
    fields = [
        FieldSchema("faq_id",          DataType.VARCHAR, max_length=100,   is_primary=True),
        FieldSchema("question",        DataType.VARCHAR, max_length=65000),
        FieldSchema("answer",          DataType.VARCHAR, max_length=65000),
        FieldSchema("question_vector", DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="FAQ question embeddings 768D")
    col = Collection(name=name, schema=schema)
    col.create_index(**hnsw_index("question_vector"))
    col.load()
    logger.info(f"  Schema faq_embeddings: faq_id | question | answer | question_vector({dim}D)")
    logger.info("  Index: HNSW M=16 efConstruction=200 COSINE")
    return col


# ============================================================================
# COLLECTION 3: document_urls  (unchanged schema)
# ============================================================================

def create_document_urls(dim: int = 768):
    from pymilvus import Collection, CollectionSchema, DataType, FieldSchema

    name = "document_urls"
    fields = [
        FieldSchema("document_id",     DataType.VARCHAR, max_length=100,  is_primary=True),
        FieldSchema("url",             DataType.VARCHAR, max_length=500),
        FieldSchema("filename",        DataType.VARCHAR, max_length=200),
        FieldSchema("file_type",       DataType.VARCHAR, max_length=20),
        FieldSchema("filename_vector", DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(
        fields,
        description="Document URLs + filename embeddings for semantic search",
    )
    col = Collection(name=name, schema=schema)
    col.create_index(**ivf_index("filename_vector"))
    col.load()
    logger.info(f"  Schema document_urls: document_id | url | filename | file_type | filename_vector({dim}D)")
    logger.info("  Index: IVF_FLAT nlist=128 COSINE")
    return col


# ============================================================================
# CREATOR MAP
# ============================================================================

CREATORS = {
    "document_chunks": create_document_chunks,
    "faq_embeddings":  create_faq_embeddings,
    "document_urls":   create_document_urls,
}


# ============================================================================
# MAIN LOGIC
# ============================================================================

def _print_stats(name: str) -> None:
    try:
        from pymilvus import Collection
        col = Collection(name)
        col.load()
        logger.info(f"  entities : {col.num_entities}")
        for idx in col.indexes:
            logger.info(f"  index    : {idx.field_name} → {idx.params}")
    except Exception as e:
        logger.debug(f"  stats unavailable: {e}")


def create_one(name: str, drop: bool, dim: int = 768) -> bool:
    logger.info(f"{'─'*55}")
    logger.info(f"📦 Collection: {name}")

    if collection_exists(name):
        if drop:
            drop_collection(name)
        else:
            logger.info("  ⚠️  Already exists — skipping (use --drop-existing to recreate)")
            _print_stats(name)
            return False

    creator = CREATORS.get(name)
    if creator is None:
        logger.error(f"  ❌ Unknown collection: {name}")
        return False

    creator(dim=dim)
    logger.info(f"  ✅ Created successfully")
    _print_stats(name)
    return True


def run(host: str, port: str, drop_existing: bool, target: Optional[str], dim: int) -> int:
    connect(host, port)

    names = [target] if target else [c.name for c in COLLECTIONS]
    created, skipped, failed = 0, 0, 0

    for name in names:
        try:
            ok = create_one(name, drop=drop_existing, dim=dim)
            created += 1 if ok else 0
            skipped += 0 if ok else 1
        except Exception as e:
            logger.error(f"  ❌ Failed {name}: {e}", exc_info=True)
            failed += 1

    logger.info(f"{'═'*55}")
    logger.info(f"📊 Summary: created={created}  skipped={skipped}  failed={failed}")

    try:
        from pymilvus import utility
        logger.info(f"📋 All collections: {utility.list_collections()}")
    except Exception:
        pass

    return 0 if failed == 0 else 1


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create Milvus collections for Vietnamese Document Pipeline v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_collections.py --host 10.22.14.6 --port 19532
  python create_collections.py --host 10.22.14.6 --port 19532 --drop-existing
  python create_collections.py --collection document_chunks --drop-existing
  python create_collections.py --collection faq_embeddings
""",
    )
    parser.add_argument("--host",          default=os.getenv("MILVUS_HOST", "10.22.14.6"))
    parser.add_argument("--port",          default=os.getenv("MILVUS_PORT", "19532"))
    parser.add_argument("--drop-existing", action="store_true")
    parser.add_argument("--collection",    choices=list(CREATORS.keys()), default=None)
    parser.add_argument("--dim",           type=int, default=768)
    args = parser.parse_args()

    try:
        return run(
            host=args.host,
            port=args.port,
            drop_existing=args.drop_existing,
            target=args.collection,
            dim=args.dim,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())