# RAG_Core/database/milvus_client.py
"""
MilvusClient v3 — tương thích schema `document_chunks` (Embedding_vectorDB v3)

Khác biệt so với bản cũ:
  - Vector field: content_vector  (trước là description_vector)
  - output_fields đầy đủ metadata: content, content_with_ctx, section_path,
    context_header, chunk_type, page_num, has_table, has_image, has_heading
  - search_documents() trả "content"/"content_with_ctx" thay "description"
"""

from pymilvus import connections, Collection, utility, db
from typing import List, Dict, Any
import numpy as np
from config.settings import settings
import logging
import time

logger = logging.getLogger(__name__)


class MilvusClient:

    DOC_VECTOR_FIELD = settings.DOCUMENT_VECTOR_FIELD      # "content_vector"
    DOC_OUTPUT_FIELDS = [
        "document_id", "content", "content_with_ctx", "section_path",
        "context_header", "chunk_type", "page_num",
        "has_table", "has_image", "has_heading",
    ]

    FAQ_VECTOR_FIELD = settings.FAQ_VECTOR_FIELD           # "question_vector"
    FAQ_OUTPUT_FIELDS = ["faq_id", "question", "answer"]

    def __init__(self):
        self.connected = False
        self.collections_cache = {}
        self._connect()

    def _connect(self):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                try:
                    connections.disconnect("default")
                except Exception:
                    pass

                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                    timeout=10
                )
                logger.info(f"✅ Connected to Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

                try:
                    db.using_database("default")
                except Exception as db_err:
                    logger.warning(f"Could not switch database: {db_err}")

                self.connected = True

                try:
                    self._load_collections()
                except Exception as load_err:
                    logger.warning(f"Collection loading failed (non-fatal): {load_err}")

                return

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    logger.error(f"Failed to connect to Milvus after {max_retries} attempts")
                    self.connected = False

    def _load_collections(self):
        try:
            available_collections = utility.list_collections()
            logger.info(f"📚 Available collections: {available_collections}")

            if settings.DOCUMENT_COLLECTION in available_collections:
                try:
                    collection = Collection(settings.DOCUMENT_COLLECTION)
                    collection.load()
                    self.collections_cache[settings.DOCUMENT_COLLECTION] = collection
                    logger.info(f"✅ Loaded: {settings.DOCUMENT_COLLECTION} ({collection.num_entities} entities)")
                except Exception as e:
                    logger.warning(f"Could not load {settings.DOCUMENT_COLLECTION}: {e}")

            if settings.FAQ_COLLECTION in available_collections:
                try:
                    collection = Collection(settings.FAQ_COLLECTION)
                    collection.load()
                    self.collections_cache[settings.FAQ_COLLECTION] = collection
                    logger.info(f"✅ Loaded: {settings.FAQ_COLLECTION} ({collection.num_entities} entities)")
                except Exception as e:
                    logger.warning(f"Could not load {settings.FAQ_COLLECTION}: {e}")

        except Exception as e:
            logger.error(f"Error loading collections: {e}")

    def check_connection(self) -> bool:
        if not self.connected:
            return False
        try:
            utility.list_collections(timeout=2)
            return True
        except Exception:
            logger.warning("Connection lost, reconnecting...")
            self._connect()
            return self.connected

    def _get_collection(self, collection_name: str) -> Collection:
        if collection_name in self.collections_cache:
            collection = self.collections_cache[collection_name]
            try:
                _ = collection.num_entities
                return collection
            except Exception:
                logger.warning(f"Cached collection {collection_name} is stale, reloading...")
                del self.collections_cache[collection_name]

        try:
            if not utility.has_collection(collection_name):
                raise ValueError(f"Collection '{collection_name}' does not exist")

            collection = Collection(collection_name)
            collection.load()
            self.collections_cache[collection_name] = collection
            logger.info(f"✅ Loaded collection: {collection_name}")
            return collection

        except Exception as e:
            logger.error(f"Failed to load collection {collection_name}: {e}")
            raise

    def _get_collection_dimension(self, collection_name: str, vector_field: str) -> int:
        try:
            collection = self._get_collection(collection_name)
            for field in collection.schema.fields:
                if field.name == vector_field:
                    return field.params.get('dim', 0)
            logger.warning(f"Vector field {vector_field} not found in {collection_name}")
            return 0
        except Exception as e:
            logger.error(f"Error getting dimension: {e}")
            return 0

    def _validate_vector_dimension(self, vector: np.ndarray, collection_name: str, vector_field: str,
                                    auto_fix: bool = True) -> np.ndarray:
        expected_dim = self._get_collection_dimension(collection_name, vector_field)
        actual_dim = vector.shape[0] if vector.ndim == 1 else vector.shape[1]

        if expected_dim == 0:
            logger.warning("Could not determine dimension, using vector as-is")
            return vector

        if actual_dim != expected_dim:
            if auto_fix:
                logger.warning(f"Dimension mismatch: expected {expected_dim}, got {actual_dim}. Auto-fixing...")
                return self._adjust_vector_dimension(vector, expected_dim)
            raise ValueError(f"Dimension mismatch: expected {expected_dim}, got {actual_dim}")

        return vector

    def _adjust_vector_dimension(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        if vector.ndim > 1:
            current_dim = vector.shape[1]
            if current_dim < target_dim:
                padding = np.zeros((vector.shape[0], target_dim - current_dim), dtype=vector.dtype)
                return np.concatenate([vector, padding], axis=1)
            elif current_dim > target_dim:
                return vector[:, :target_dim]
        else:
            current_dim = vector.shape[0]
            if current_dim < target_dim:
                padding = np.zeros(target_dim - current_dim, dtype=vector.dtype)
                return np.concatenate([vector, padding])
            elif current_dim > target_dim:
                return vector[:target_dim]
        return vector

    # ================================================================
    # SEARCH — document_chunks (schema mới)
    # ================================================================

    def search_documents(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        max_retries = 2

        for attempt in range(max_retries):
            try:
                if not self.check_connection():
                    raise ConnectionError("Not connected to Milvus")

                collection = self._get_collection(settings.DOCUMENT_COLLECTION)

                query_vector = self._validate_vector_dimension(
                    query_vector, settings.DOCUMENT_COLLECTION, self.DOC_VECTOR_FIELD
                )

                search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

                results = collection.search(
                    data=[query_vector.tolist()],
                    anns_field=self.DOC_VECTOR_FIELD,
                    param=search_params,
                    limit=top_k,
                    output_fields=self.DOC_OUTPUT_FIELDS
                )

                documents = []
                for hits in results:
                    for hit in hits:
                        documents.append({
                            "document_id":     hit.entity.get("document_id"),
                            "content":          hit.entity.get("content"),
                            "content_with_ctx": hit.entity.get("content_with_ctx"),
                            "section_path":     hit.entity.get("section_path"),
                            "context_header":   hit.entity.get("context_header"),
                            "chunk_type":       hit.entity.get("chunk_type"),
                            "page_num":         hit.entity.get("page_num"),
                            "has_table":        bool(hit.entity.get("has_table")),
                            "has_image":        bool(hit.entity.get("has_image")),
                            "has_heading":      bool(hit.entity.get("has_heading")),
                            "similarity_score": hit.score,
                        })

                logger.info(f"✅ Found {len(documents)} chunks")
                return documents

            except Exception as e:
                logger.error(f"Search attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    self._connect()
                    time.sleep(1)
                else:
                    raise

    def search_faq(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        max_retries = 2

        for attempt in range(max_retries):
            try:
                if not self.check_connection():
                    raise ConnectionError("Not connected to Milvus")

                collection = self._get_collection(settings.FAQ_COLLECTION)

                query_vector = self._validate_vector_dimension(
                    query_vector, settings.FAQ_COLLECTION, self.FAQ_VECTOR_FIELD
                )

                search_params = {"metric_type": "COSINE", "params": {"ef": 64}}

                results = collection.search(
                    data=[query_vector.tolist()],
                    anns_field=self.FAQ_VECTOR_FIELD,
                    param=search_params,
                    limit=top_k,
                    output_fields=self.FAQ_OUTPUT_FIELDS
                )

                faqs = []
                for hits in results:
                    for hit in hits:
                        faqs.append({
                            "faq_id": hit.entity.get("faq_id"),
                            "question": hit.entity.get("question"),
                            "answer": hit.entity.get("answer"),
                            "similarity_score": hit.score,
                        })

                logger.info(f"✅ Found {len(faqs)} FAQs")
                return faqs

            except Exception as e:
                logger.error(f"FAQ search attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    self._connect()
                    time.sleep(1)
                else:
                    raise

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        try:
            if not utility.has_collection(collection_name):
                return {"error": f"Collection {collection_name} does not exist"}

            collection = Collection(collection_name)
            schema = collection.schema

            fields_info = [
                {"name": f.name, "dtype": str(f.dtype), "params": f.params}
                for f in schema.fields
            ]

            return {
                "collection_name": collection_name,
                "fields": fields_info,
                "description": schema.description,
                "num_entities": collection.num_entities
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}


# Global instance
milvus_client = MilvusClient()