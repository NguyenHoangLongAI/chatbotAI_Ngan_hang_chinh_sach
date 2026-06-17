#!/usr/bin/env python3
"""
milvus_client.py — MilvusManager v2
=====================================
Thay đổi:
  - insert_chunks() thay insert_embeddings() → dùng collection document_chunks
  - delete_document() xóa từ document_chunks (trước là document_embeddings)
  - Tương thích ngược: insert_embeddings() gọi insert_chunks() nội bộ
"""

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import List, Dict, Any, Optional
import asyncio
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MilvusManager:

    def __init__(self, host="10.22.14.6", port="19532", embedding_dim=768):
        self.host = host
        self.port = port
        self.embedding_dim = embedding_dim

        # Collection names
        self.chunks_collection_name = "document_chunks"    # v2 — full metadata
        self.faq_collection_name    = "faq_embeddings"

        self._chunks_col:  Optional[Collection] = None
        self._faq_col:     Optional[Collection] = None
        self.is_initialized = False

        # Field length limits (mirror schema)
        self._max_id           = 218
        self._max_document_id  = 98
        self._max_content      = 7990
        self._max_ctx          = 9990
        self._max_section      = 498
        self._max_ctx_header   = 998
        self._max_chunk_type   = 28
        self._max_question     = 60000
        self._max_answer       = 60000

    # ================================================================
    # INIT / CONNECT
    # ================================================================

    async def initialize(self, max_retries: int = 5, retry_delay: int = 2):
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Connecting to Milvus {self.host}:{self.port} "
                    f"(attempt {attempt+1}/{max_retries})"
                )
                try:
                    connections.disconnect("default")
                except Exception:
                    pass
                connections.connect("default", host=self.host, port=int(self.port))
                logger.info(f"✅ Connected to Milvus")

                await self._ensure_chunks_collection()
                await self._ensure_faq_collection()

                self.is_initialized = True
                logger.info("✅ Milvus initialization complete")
                return True

            except Exception as e:
                logger.error(f"❌ Milvus init error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    self.is_initialized = False
                    raise

    def _check_initialized(self):
        if not self.is_initialized:
            raise RuntimeError("Milvus is not initialized. Call initialize() first.")

    # ================================================================
    # COLLECTION SETUP
    # ================================================================

    async def _ensure_chunks_collection(self):
        """Load document_chunks if exists; create otherwise."""
        name = self.chunks_collection_name
        try:
            if utility.has_collection(name):
                logger.info(f"📦 {name} already exists — loading")
                self._chunks_col = Collection(name)
                self._chunks_col.load()
                logger.info(f"✅ Loaded {name} ({self._chunks_col.num_entities} entities)")
                return

            logger.info(f"Creating collection: {name}")
            dim = self.embedding_dim
            fields = [
                FieldSchema("id",               DataType.VARCHAR, max_length=220,   is_primary=True),
                FieldSchema("document_id",      DataType.VARCHAR, max_length=100),
                FieldSchema("content",          DataType.VARCHAR, max_length=8000),
                FieldSchema("content_with_ctx", DataType.VARCHAR, max_length=10000),
                FieldSchema("section_path",     DataType.VARCHAR, max_length=500),
                FieldSchema("context_header",   DataType.VARCHAR, max_length=1000),
                FieldSchema("chunk_type",       DataType.VARCHAR, max_length=30),
                FieldSchema("page_num",         DataType.INT32),
                FieldSchema("part_index",       DataType.INT16),
                FieldSchema("total_parts",      DataType.INT16),
                FieldSchema("token_count",      DataType.INT32),
                FieldSchema("char_count",       DataType.INT32),
                FieldSchema("has_table",        DataType.INT8),
                FieldSchema("has_image",        DataType.INT8),
                FieldSchema("has_heading",      DataType.INT8),
                FieldSchema("is_overlap",       DataType.INT8),
                FieldSchema("content_vector",   DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(fields, description="Document chunks — full metadata 768D")
            self._chunks_col = Collection(name=name, schema=schema)
            self._chunks_col.create_index(
                field_name="content_vector",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            self._chunks_col.load()
            logger.info(f"✅ Created {name} with HNSW index")

        except Exception as e:
            logger.error(f"❌ {name} setup error: {e}")
            raise

    async def _ensure_faq_collection(self):
        name = self.faq_collection_name
        try:
            if utility.has_collection(name):
                self._faq_col = Collection(name)
                self._faq_col.load()
                logger.info(f"✅ Loaded {name}")
                return

            fields = [
                FieldSchema("faq_id",          DataType.VARCHAR, max_length=100,  is_primary=True),
                FieldSchema("question",        DataType.VARCHAR, max_length=65000),
                FieldSchema("answer",          DataType.VARCHAR, max_length=65000),
                FieldSchema("question_vector", DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ]
            schema = CollectionSchema(fields, description="FAQ embeddings 768D")
            self._faq_col = Collection(name=name, schema=schema)
            self._faq_col.create_index(
                field_name="question_vector",
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "HNSW",
                    "params": {"M": 16, "efConstruction": 200},
                },
            )
            self._faq_col.load()
            logger.info(f"✅ Created {name}")
        except Exception as e:
            logger.error(f"❌ {name} setup error: {e}")
            raise

    # ================================================================
    # INSERT CHUNKS (primary method — v2)
    # ================================================================

    async def insert_chunks(self, records) -> int:
        """
        Insert List[ChunkRecord] vào document_chunks.
        records: List[ChunkRecord] (từ document_processor.py)
        Trả về số record đã lưu thành công.
        """
        self._check_initialized()
        if not self._chunks_col or not records:
            return 0

        try:
            self._chunks_col.load()
        except Exception:
            pass

        # Validate + build entity lists
        valid = []
        for rec in records:
            vec = rec.content_vector
            if not vec or len(vec) != self.embedding_dim:
                logger.warning(f"  Skip chunk {rec.chunk_index}: invalid vector")
                continue
            d = rec.to_milvus_dict()
            # Trim strings to schema limits
            d["id"]               = d["id"][:220]
            d["document_id"]      = d["document_id"][:100]
            d["content"]          = d["content"][:8000]
            d["content_with_ctx"] = d["content_with_ctx"][:10000]
            d["section_path"]     = d["section_path"][:500]
            d["context_header"]   = d["context_header"][:1000]
            d["chunk_type"]       = d["chunk_type"][:30]
            valid.append(d)

        if not valid:
            logger.warning("No valid chunks to insert")
            return 0

        total_inserted = 0
        batch_size = 50

        for start in range(0, len(valid), batch_size):
            batch = valid[start: start + batch_size]
            entities = [
                [x["id"]               for x in batch],
                [x["document_id"]      for x in batch],
                [x["content"]          for x in batch],
                [x["content_with_ctx"] for x in batch],
                [x["section_path"]     for x in batch],
                [x["context_header"]   for x in batch],
                [x["chunk_type"]       for x in batch],
                [x["page_num"]         for x in batch],
                [x["part_index"]       for x in batch],
                [x["total_parts"]      for x in batch],
                [x["token_count"]      for x in batch],
                [x["char_count"]       for x in batch],
                [x["has_table"]        for x in batch],
                [x["has_image"]        for x in batch],
                [x["has_heading"]      for x in batch],
                [x["is_overlap"]       for x in batch],
                [x["content_vector"]   for x in batch],
            ]
            try:
                self._chunks_col.insert(entities)
                total_inserted += len(batch)
                batch_num = start // batch_size + 1
                total_batches = (len(valid) + batch_size - 1) // batch_size
                logger.info(f"  Batch {batch_num}/{total_batches}: {total_inserted} inserted")
            except Exception as e:
                logger.error(f"  Batch insert error: {e}")
                continue

        self._chunks_col.flush()
        logger.info(f"✅ Total inserted into document_chunks: {total_inserted}")
        return total_inserted

    # ── Backward compat: old insert_embeddings() API ─────────────────

    async def insert_embeddings(self, embeddings_data: List[Dict]) -> int:
        """
        Deprecated — kept for backward compatibility.
        Converts old dict format to document_chunks format.
        """
        logger.warning("insert_embeddings() is deprecated — use insert_chunks()")
        # Build minimal ChunkRecord-like dicts for direct insert
        if not self._chunks_col or not embeddings_data:
            return 0
        try:
            self._chunks_col.load()
        except Exception:
            pass

        valid = []
        for item in embeddings_data:
            vec = item.get("description_vector", [])
            if len(vec) != self.embedding_dim:
                continue
            content = item.get("description", "")[:8000]
            doc_id  = item.get("document_id", "")[:100]
            chunk_id = item.get("id", "")[:220]
            valid.append({
                "id":               chunk_id,
                "document_id":      doc_id,
                "content":          content,
                "content_with_ctx": content,
                "section_path":     "",
                "context_header":   "",
                "chunk_type":       "legacy",
                "page_num":         0,
                "part_index":       0,
                "total_parts":      1,
                "token_count":      0,
                "char_count":       len(content),
                "has_table":        0,
                "has_image":        0,
                "has_heading":      0,
                "is_overlap":       0,
                "content_vector":   vec,
            })

        if not valid:
            return 0

        entities = [
            [x["id"]               for x in valid],
            [x["document_id"]      for x in valid],
            [x["content"]          for x in valid],
            [x["content_with_ctx"] for x in valid],
            [x["section_path"]     for x in valid],
            [x["context_header"]   for x in valid],
            [x["chunk_type"]       for x in valid],
            [x["page_num"]         for x in valid],
            [x["part_index"]       for x in valid],
            [x["total_parts"]      for x in valid],
            [x["token_count"]      for x in valid],
            [x["char_count"]       for x in valid],
            [x["has_table"]        for x in valid],
            [x["has_image"]        for x in valid],
            [x["has_heading"]      for x in valid],
            [x["is_overlap"]       for x in valid],
            [x["content_vector"]   for x in valid],
        ]
        self._chunks_col.insert(entities)
        self._chunks_col.flush()
        logger.info(f"✅ insert_embeddings (legacy): {len(valid)} records")
        return len(valid)

    # ================================================================
    # DELETE
    # ================================================================

    async def delete_document(self, document_id: str) -> bool:
        try:
            self._check_initialized()
            expr = f'document_id == "{document_id}"'
            self._chunks_col.delete(expr)
            self._chunks_col.flush()
            logger.info(f"✅ Deleted chunks for document_id: {document_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Delete document error: {e}")
            return False

    # ================================================================
    # FAQ
    # ================================================================

    async def insert_faq(
        self, faq_id: str, question: str, answer: str, question_vector: List[float]
    ) -> bool:
        try:
            self._check_initialized()
            if not self._faq_col:
                return False
            try:
                self._faq_col.load()
            except Exception:
                pass

            faq_id   = faq_id[:100]
            question = question[:self._max_question]
            answer   = answer[:self._max_answer]

            if len(question_vector) != self.embedding_dim:
                return False

            self._faq_col.insert([[faq_id], [question], [answer], [question_vector]])
            self._faq_col.flush()
            logger.info(f"✅ Inserted FAQ: {faq_id}")
            return True
        except Exception as e:
            logger.error(f"❌ FAQ insert error: {e}")
            return False

    async def delete_faq(self, faq_id: str) -> bool:
        try:
            self._check_initialized()
            self._faq_col.delete(f'faq_id == "{faq_id}"')
            self._faq_col.flush()
            logger.info(f"✅ Deleted FAQ: {faq_id}")
            return True
        except Exception as e:
            logger.error(f"❌ FAQ delete error: {e}")
            return False

    # ================================================================
    # HEALTH
    # ================================================================

    async def health_check(self) -> bool:
        try:
            if not self.is_initialized:
                return False
            connections.get_connection_addr("default")
            return True
        except Exception:
            return False