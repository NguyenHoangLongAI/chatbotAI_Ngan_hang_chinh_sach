from pymilvus import connections, Collection, utility, db
from typing import List, Dict, Any
import numpy as np
from config.settings import settings
import logging
import time

logger = logging.getLogger(__name__)


class MilvusClient:
    def __init__(self):
        self.connected = False
        self.expected_dimension = None
        self.collections_cache = {}
        self._connect()

    def _connect(self):
        """Connect to Milvus with retry"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Disconnect existing
                try:
                    connections.disconnect("default")
                except:
                    pass

                # Connect
                connections.connect(
                    alias="default",
                    host=settings.MILVUS_HOST,
                    port=settings.MILVUS_PORT,
                    timeout=10
                )

                logger.info(f"✅ Connected to Milvus: {settings.MILVUS_HOST}:{settings.MILVUS_PORT}")

                # Switch to default database
                try:
                    db.using_database("default")
                    logger.info(f"✅ Using database: default")
                except Exception as db_err:
                    logger.warning(f"Could not switch database: {db_err}")

                self.connected = True

                # Load collections (but don't fail if this errors)
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
        """Load collections with error handling"""
        try:
            available_collections = utility.list_collections()
            logger.info(f"📚 Available collections: {available_collections}")

            # Load document collection
            if settings.DOCUMENT_COLLECTION in available_collections:
                try:
                    collection = Collection(settings.DOCUMENT_COLLECTION)
                    collection.load()
                    self.collections_cache[settings.DOCUMENT_COLLECTION] = collection
                    logger.info(f"✅ Loaded: {settings.DOCUMENT_COLLECTION} ({collection.num_entities} entities)")
                except Exception as e:
                    logger.warning(f"Could not load {settings.DOCUMENT_COLLECTION}: {e}")

            # Load FAQ collection
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
        """Check if connected - simplified version"""
        if not self.connected:
            return False

        try:
            # Simple ping
            utility.list_collections(timeout=2)
            return True
        except:
            # Connection lost, try to reconnect
            logger.warning("Connection lost, reconnecting...")
            self._connect()
            return self.connected

    def _get_collection(self, collection_name: str) -> Collection:
        """Get collection with lazy loading"""
        # Check cache first
        if collection_name in self.collections_cache:
            collection = self.collections_cache[collection_name]
            # Verify collection is still valid
            try:
                _ = collection.num_entities
                return collection
            except:
                # Cache is stale, reload
                logger.warning(f"Cached collection {collection_name} is stale, reloading...")
                del self.collections_cache[collection_name]

        # Load collection
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
        """Get dimension with error handling"""
        try:
            collection = self._get_collection(collection_name)
            schema = collection.schema
            for field in schema.fields:
                if field.name == vector_field:
                    dim = field.params.get('dim', 0)
                    logger.debug(f"Collection {collection_name}.{vector_field} dimension: {dim}")
                    return dim
            logger.warning(f"Vector field {vector_field} not found in {collection_name}")
            return 0
        except Exception as e:
            logger.error(f"Error getting dimension: {str(e)}")
            return 0

    def _validate_vector_dimension(self, vector: np.ndarray, collection_name: str, vector_field: str,
                                   auto_fix: bool = True) -> np.ndarray:
        """Validate and adjust vector dimension"""
        expected_dim = self._get_collection_dimension(collection_name, vector_field)
        actual_dim = vector.shape[0] if vector.ndim == 1 else vector.shape[1]

        if expected_dim == 0:
            logger.warning(f"Could not determine dimension, using vector as-is")
            return vector

        if actual_dim != expected_dim:
            if auto_fix:
                logger.warning(f"Dimension mismatch: expected {expected_dim}, got {actual_dim}. Auto-fixing...")
                return self._adjust_vector_dimension(vector, expected_dim)
            else:
                raise ValueError(f"Dimension mismatch: expected {expected_dim}, got {actual_dim}")

        return vector

    def _adjust_vector_dimension(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """Adjust vector dimension"""
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

    def search_documents(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents with retry logic"""
        max_retries = 2

        for attempt in range(max_retries):
            try:
                # Check connection
                if not self.check_connection():
                    raise ConnectionError("Not connected to Milvus")

                # Get collection
                collection = self._get_collection(settings.DOCUMENT_COLLECTION)

                # Validate vector
                query_vector = self._validate_vector_dimension(
                    query_vector, settings.DOCUMENT_COLLECTION, "description_vector"
                )

                # Search
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"ef": 64}
                }

                results = collection.search(
                    data=[query_vector.tolist()],
                    anns_field="description_vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=["document_id", "description"]
                )

                documents = []
                for hits in results:
                    for hit in hits:
                        documents.append({
                            "document_id": hit.entity.get("document_id"),
                            "description": hit.entity.get("description"),
                            "similarity_score": hit.score
                        })

                logger.info(f"✅ Found {len(documents)} documents")
                return documents

            except Exception as e:
                logger.error(f"Search attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    # Retry: reconnect
                    self._connect()
                    time.sleep(1)
                else:
                    raise

    def search_faq(self, query_vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search FAQ with retry logic"""
        max_retries = 2

        for attempt in range(max_retries):
            try:
                # Check connection
                if not self.check_connection():
                    raise ConnectionError("Not connected to Milvus")

                # Get collection
                collection = self._get_collection(settings.FAQ_COLLECTION)

                # Validate vector
                query_vector = self._validate_vector_dimension(
                    query_vector, settings.FAQ_COLLECTION, "question_vector"
                )

                # Search
                search_params = {
                    "metric_type": "COSINE",
                    "params": {"ef": 64}
                }

                results = collection.search(
                    data=[query_vector.tolist()],
                    anns_field="question_vector",
                    param=search_params,
                    limit=top_k,
                    output_fields=["faq_id", "question", "answer"]
                )

                faqs = []
                for hits in results:
                    for hit in hits:
                        faqs.append({
                            "faq_id": hit.entity.get("faq_id"),
                            "question": hit.entity.get("question"),
                            "answer": hit.entity.get("answer"),
                            "similarity_score": hit.score
                        })

                logger.info(f"✅ Found {len(faqs)} FAQs")
                return faqs

            except Exception as e:
                logger.error(f"FAQ search attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    # Retry: reconnect
                    self._connect()
                    time.sleep(1)
                else:
                    raise

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get collection info"""
        try:
            if not utility.has_collection(collection_name):
                return {"error": f"Collection {collection_name} does not exist"}

            collection = Collection(collection_name)
            schema = collection.schema

            fields_info = []
            for field in schema.fields:
                field_info = {
                    "name": field.name,
                    "dtype": str(field.dtype),
                    "params": field.params
                }
                fields_info.append(field_info)

            return {
                "collection_name": collection_name,
                "fields": fields_info,
                "description": schema.description,
                "num_entities": collection.num_entities
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {"error": str(e)}


# Global instance
milvus_client = MilvusClient()