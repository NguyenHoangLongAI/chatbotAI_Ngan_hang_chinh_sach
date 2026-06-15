# Embedding_vectorDB/document_urls_collection.py
"""
Script to create and manage document_urls collection
Maps document_id to public URLs with filename embedding for search
"""

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)
import os
import logging
import numpy as np
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentURLsManager:
    """Manager for document URLs collection with filename embedding"""

    def __init__(
            self,
            host: str = "localhost",
            port: str = "19530",
            collection_name: str = "document_urls",
            embedding_dim: int = 768  # Vietnamese SBERT dimension
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.collection = None

        # Connect to Milvus
        self._connect()

        # Initialize embedding model (lazy loading)
        self._embedding_model = None

    def _connect(self):
        """Connect to Milvus"""
        try:
            # Auto-detect if running inside Docker
            if self.host == "milvus":
                import socket
                try:
                    socket.gethostbyname("milvus")
                except socket.gaierror:
                    self.host = "localhost"
                    logger.warning("⚠️ Running outside Docker, using localhost")

            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            logger.info(f"✅ Connected to Milvus at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            raise

    @property
    def embedding_model(self):
        """Lazy load embedding model (FORCE CPU)"""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                # ✅ FIX: Force CPU to avoid CUDA compatibility issues
                logger.info("🔄 Loading Vietnamese SBERT model (CPU mode)...")

                # Force CPU
                device = 'cpu'

                self._embedding_model = SentenceTransformer(
                    'keepitreal/vietnamese-sbert',
                    device=device
                )

                logger.info(f"✅ Embedding model loaded on {device}")

            except ImportError:
                logger.error("❌ sentence-transformers not installed")
                raise ImportError("Run: pip install sentence-transformers")
            except Exception as e:
                logger.error(f"❌ Failed to load embedding model: {e}")
                raise

        return self._embedding_model

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            if not text or not text.strip():
                # Return zero vector for empty text
                return [0.0] * self.embedding_dim

            embedding = self.embedding_model.encode(
                text.strip(),
                normalize_embeddings=True
            )

            return embedding.tolist()

        except Exception as e:
            logger.error(f"❌ Embedding error: {e}")
            return [0.0] * self.embedding_dim

    def create_collection(self):
        """Create document_urls collection with filename embedding"""
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists")
                self.collection = Collection(self.collection_name)
                self.collection.load()
                return

            # Define schema with vector field
            fields = [
                FieldSchema(
                    name="document_id",
                    dtype=DataType.VARCHAR,
                    max_length=100,
                    is_primary=True,
                    description="Unique document identifier"
                ),
                FieldSchema(
                    name="url",
                    dtype=DataType.VARCHAR,
                    max_length=500,
                    description="Public URL to the document"
                ),
                FieldSchema(
                    name="filename",
                    dtype=DataType.VARCHAR,
                    max_length=200,
                    description="Original filename"
                ),
                FieldSchema(
                    name="file_type",
                    dtype=DataType.VARCHAR,
                    max_length=20,
                    description="File extension (.pdf, .docx, etc.)"
                ),
                # ✅ NEW: Vector field for filename embedding
                FieldSchema(
                    name="filename_vector",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.embedding_dim,
                    description="Embedding of filename for semantic search"
                )
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Document URLs with filename embeddings for semantic search"
            )

            # Create collection
            self.collection = Collection(
                name=self.collection_name,
                schema=schema,
                using='default'
            )

            # Create index on filename_vector
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }

            self.collection.create_index(
                field_name="filename_vector",
                index_params=index_params
            )

            # Load collection
            self.collection.load()

            logger.info(f"✅ Collection '{self.collection_name}' created successfully")

        except Exception as e:
            logger.error(f"❌ Error creating collection: {e}")
            raise

    def insert_url(
            self,
            document_id: str,
            url: str,
            filename: str = "",
            file_type: str = ""
    ) -> bool:
        """
        Insert or update document URL with filename embedding

        Args:
            document_id: Document identifier
            url: Public URL
            filename: Original filename (will be embedded)
            file_type: File extension
        """
        try:
            if not self.collection:
                raise Exception("Collection not initialized")

            # Validate inputs
            if len(document_id) > 100:
                document_id = document_id[:100]
            if len(url) > 500:
                logger.warning(f"URL too long, truncating: {url[:50]}...")
                url = url[:500]
            if len(filename) > 200:
                filename = filename[:200]
            if len(file_type) > 20:
                file_type = file_type[:20]

            # Generate filename embedding
            logger.info(f"🔄 Embedding filename: {filename}")
            filename_embedding = self.embed_text(filename)

            # ✅ FIX: Don't delete with VARCHAR pk - use expr with IN
            # Delete existing entry if any (use proper expression)
            try:
                # Query to check if exists first
                expr = f'document_id == "{document_id}"'
                existing = self.collection.query(
                    expr=expr,
                    output_fields=["document_id"],
                    limit=1
                )

                if existing:
                    # Delete using pk list instead of expression
                    pks = [document_id]
                    self.collection.delete(expr=f'document_id in {pks}')
                    logger.debug(f"Deleted existing entry for {document_id}")
            except Exception as del_error:
                # If delete fails, just log and continue
                logger.debug(f"No existing entry or delete failed: {del_error}")

            # Insert new entry
            entities = [
                [document_id],
                [url],
                [filename],
                [file_type],
                [filename_embedding]
            ]

            self.collection.insert(entities)
            self.collection.flush()

            logger.info(f"✅ Inserted URL for document: {document_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Error inserting URL: {e}")
            return False

    def batch_insert(self, documents: list) -> int:
        """
        Batch insert document URLs with filename embeddings

        Args:
            documents: List of dicts with keys: document_id, url, filename, file_type

        Returns:
            Number of successfully inserted documents
        """
        try:
            if not documents:
                return 0

            document_ids = []
            urls = []
            filenames = []
            file_types = []
            filename_vectors = []

            logger.info(f"🔄 Generating embeddings for {len(documents)} filenames...")

            for doc in documents:
                document_id = doc.get('document_id', '')[:100]
                url = doc.get('url', '')[:500]
                filename = doc.get('filename', '')[:200]
                file_type = doc.get('file_type', '')[:20]

                if document_id and url:
                    document_ids.append(document_id)
                    urls.append(url)
                    filenames.append(filename)
                    file_types.append(file_type)

                    # Generate embedding
                    filename_vector = self.embed_text(filename)
                    filename_vectors.append(filename_vector)

            if not document_ids:
                logger.warning("No valid documents to insert")
                return 0

            # ✅ FIX: Improved delete logic for batch
            # Delete existing entries one by one to avoid expr issues
            logger.info(f"🔄 Checking for existing entries...")
            deleted_count = 0

            for doc_id in document_ids:
                try:
                    expr = f'document_id == "{doc_id}"'
                    existing = self.collection.query(
                        expr=expr,
                        output_fields=["document_id"],
                        limit=1
                    )

                    if existing:
                        # Use IN operator with list
                        self.collection.delete(expr=f'document_id in ["{doc_id}"]')
                        deleted_count += 1
                except Exception as e:
                    logger.debug(f"Could not delete {doc_id}: {e}")
                    continue

            if deleted_count > 0:
                logger.info(f"✅ Deleted {deleted_count} existing entries")

            # Batch insert
            entities = [
                document_ids,
                urls,
                filenames,
                file_types,
                filename_vectors
            ]

            self.collection.insert(entities)
            self.collection.flush()

            logger.info(f"✅ Batch inserted {len(document_ids)} URLs with embeddings")
            return len(document_ids)

        except Exception as e:
            logger.error(f"❌ Batch insert error: {e}")
            return 0

    def search_by_filename(
            self,
            query: str,
            top_k: int = 5,
            min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Search documents by filename using semantic search

        Args:
            query: Search query (e.g., "hướng dẫn sử dụng")
            top_k: Number of results
            min_score: Minimum similarity score

        Returns:
            List of documents with URLs
        """
        try:
            # Generate query embedding
            query_vector = self.embed_text(query)

            # Search
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}
            }

            results = self.collection.search(
                data=[query_vector],
                anns_field="filename_vector",
                param=search_params,
                limit=top_k,
                output_fields=["document_id", "url", "filename", "file_type"]
            )

            # Format results
            documents = []
            for hits in results:
                for hit in hits:
                    if hit.score >= min_score:
                        documents.append({
                            "document_id": hit.entity.get("document_id"),
                            "url": hit.entity.get("url"),
                            "filename": hit.entity.get("filename"),
                            "file_type": hit.entity.get("file_type"),
                            "similarity_score": hit.score
                        })

            logger.info(f"✅ Found {len(documents)} documents for query: {query}")
            return documents

        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    def get_url(self, document_id: str) -> dict:
        """
        Get URL for a document

        Returns:
            Dict with url, filename, file_type or None
        """
        try:
            expr = f'document_id == "{document_id}"'

            results = self.collection.query(
                expr=expr,
                output_fields=["url", "filename", "file_type"],
                limit=1
            )

            if results:
                return {
                    "document_id": document_id,
                    "url": results[0].get("url"),
                    "filename": results[0].get("filename"),
                    "file_type": results[0].get("file_type")
                }

            return None

        except Exception as e:
            logger.error(f"❌ Error getting URL: {e}")
            return None

    def batch_get_urls(self, document_ids: list) -> dict:
        """
        Get URLs for multiple documents

        Returns:
            Dict mapping document_id -> url info
        """
        try:
            results = {}

            for doc_id in document_ids:
                url_info = self.get_url(doc_id)
                if url_info:
                    results[doc_id] = url_info

            return results

        except Exception as e:
            logger.error(f"❌ Batch get error: {e}")
            return {}

    def delete_url(self, document_id: str) -> bool:
        """Delete URL entry"""
        try:
            # ✅ FIX: Use IN operator for VARCHAR pk
            expr = f'document_id in ["{document_id}"]'
            self.collection.delete(expr)
            self.collection.flush()
            logger.info(f"✅ Deleted URL for: {document_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Delete error: {e}")
            return False

    def count_documents(self) -> int:
        """Get total number of document URLs"""
        try:
            self.collection.load()
            return self.collection.num_entities
        except Exception as e:
            logger.error(f"❌ Count error: {e}")
            return 0

    def list_all_urls(self, limit: int = 100) -> list:
        """List all document URLs"""
        try:
            results = self.collection.query(
                expr="document_id != ''",
                output_fields=["document_id", "url", "filename", "file_type"],
                limit=limit
            )

            return results

        except Exception as e:
            logger.error(f"❌ List error: {e}")
            return []


def main():
    """Test the DocumentURLsManager"""

    # Initialize manager
    manager = DocumentURLsManager(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=os.getenv("MILVUS_PORT", "19530")
    )

    # Create collection
    manager.create_collection()

if __name__ == "__main__":
    main()