# RAG_Core/services/document_url_service.py - NEW FILE

from typing import Dict, List, Optional, Any
from database.milvus_client import milvus_client
from config.settings import settings
from pymilvus import Collection, utility
import logging

logger = logging.getLogger(__name__)


class DocumentURLService:
    """
    Service để fetch document URLs từ Milvus collection
    và convert sang public URLs
    """

    def __init__(self):
        self.collection_name = settings.DOCUMENT_URLS_COLLECTION
        self.collection = None
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize document_urls collection"""
        try:
            if not milvus_client.check_connection():
                logger.warning("Milvus not connected, URL service disabled")
                return

            if not utility.has_collection(self.collection_name):
                logger.warning(f"Collection {self.collection_name} not found")
                return

            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info(f"✅ Document URL service initialized: {self.collection_name}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize URL service: {e}")
            self.collection = None

    def get_document_url(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document URL by document_id

        Args:
            document_id: Document ID

        Returns:
            {
                "document_id": "doc_001",
                "url": "https://ngrok.../file.pdf",
                "filename": "file.pdf",
                "file_type": ".pdf",
                "is_public": True
            }
        """
        try:
            if not self.collection:
                return None

            # Query Milvus
            expr = f'document_id == "{document_id}"'

            results = self.collection.query(
                expr=expr,
                output_fields=["document_id", "url", "filename", "file_type"],
                limit=1
            )

            if not results:
                logger.debug(f"No URL found for document_id: {document_id}")
                return None

            result = results[0]
            internal_url = result.get("url", "")

            # Convert to public URL
            public_url = settings.get_public_url(internal_url)

            return {
                "document_id": document_id,
                "url": public_url,
                "internal_url": internal_url,
                "filename": result.get("filename", ""),
                "file_type": result.get("file_type", ""),
                "is_public": public_url != internal_url
            }

        except Exception as e:
            logger.error(f"Error getting URL for {document_id}: {e}")
            return None

    def batch_get_document_urls(
            self,
            document_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get URLs for multiple documents

        Args:
            document_ids: List of document IDs

        Returns:
            {
                "doc_001": {"url": "...", "filename": "..."},
                "doc_002": {"url": "...", "filename": "..."}
            }
        """
        result = {}

        if not self.collection:
            return result

        for doc_id in document_ids:
            url_info = self.get_document_url(doc_id)
            if url_info:
                result[doc_id] = url_info

        return result

    def enrich_references_with_urls(
            self,
            references: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich references with document URLs

        Args:
            references: List of reference dicts with document_id

        Returns:
            List of enriched references with URLs
        """
        if not references:
            return []

        enriched = []

        for ref in references:
            ref_copy = ref.copy()
            document_id = ref.get("document_id")

            if document_id:
                url_info = self.get_document_url(document_id)

                if url_info:
                    ref_copy["url"] = url_info["url"]
                    ref_copy["filename"] = url_info["filename"]
                    ref_copy["file_type"] = url_info["file_type"]

                    logger.debug(
                        f"✅ Enriched {document_id}: {url_info['filename']}"
                    )

            enriched.append(ref_copy)

        return enriched


# Global instance
document_url_service = DocumentURLService()