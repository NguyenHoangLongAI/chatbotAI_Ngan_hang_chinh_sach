# RAG_Core/tools/vector_search.py
"""
vector_search.py — v4
- Thay Cohere Rerank API bằng BAAI/bge-reranker-v2-m3 (local, không call API)
- Giữ nguyên schema document_chunks (content_vector, content_with_ctx...)
- Interface các @tool không đổi — các agent dùng bình thường
"""

from langchain_core.tools import tool
from typing import List, Dict, Any
import numpy as np

from models.embedding_model import embedding_model
from models.reranker_model import reranker_model      # ← local reranker mới
from database.milvus_client import milvus_client
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER — lấy text đại diện cho một document dict
# ============================================================================

def _doc_to_text(doc: Dict[str, Any]) -> str:
    """Ưu tiên content_with_ctx (có context header), fallback về content."""
    return (
        doc.get("content_with_ctx")
        or doc.get("content")
        or doc.get("description", "")
        or ""
    )


def _faq_to_text(faq: Dict[str, Any]) -> str:
    question = faq.get("question", "").strip()
    answer = faq.get("answer", "").strip()
    return f"Câu hỏi: {question}\nTrả lời: {answer}"


# ============================================================================
# FAQ RERANKING
# ============================================================================

@tool
def rerank_faq(query: str, faq_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank FAQ bằng local cross-encoder BAAI/bge-reranker-v2-m3 (tối ưu Tiếng Việt)."""
    if not faq_results:
        return []

    if not reranker_model.available:
        logger.warning("Reranker not available, returning FAQs sorted by similarity_score")
        return sorted(faq_results, key=lambda x: x.get("similarity_score", 0), reverse=True)

    try:
        documents = [_faq_to_text(faq) for faq in faq_results]

        indexed_scores = reranker_model.rerank_with_index(
            query=query,
            documents=documents,
            max_length=settings.RERANKER_MAX_LENGTH,
        )

        reranked = []
        for orig_idx, score in indexed_scores:
            faq_copy = faq_results[orig_idx].copy()
            faq_copy["rerank_score"] = score
            faq_copy["rerank_source"] = "bge-reranker-v2-m3"
            reranked.append(faq_copy)

        logger.info(
            f"✅ Reranked {len(reranked)} FAQs. "
            f"Best score: {reranked[0].get('rerank_score', 0):.4f}"
        )
        return reranked

    except Exception as e:
        logger.error(f"❌ Error in FAQ reranking: {e}", exc_info=True)
        return sorted(faq_results, key=lambda x: x.get("similarity_score", 0), reverse=True)


# ============================================================================
# DOCUMENT RERANKING
# ============================================================================

@tool
def rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rerank document chunks bằng local cross-encoder BAAI/bge-reranker-v2-m3."""
    if not documents:
        return []

    if not reranker_model.available:
        logger.warning("Reranker not available, returning documents as-is")
        return documents

    try:
        doc_texts = [_doc_to_text(doc) for doc in documents]

        if not any(doc_texts):
            logger.warning("No valid document texts found for reranking")
            return documents

        logger.info(f"🔄 Reranking {len(doc_texts)} chunks với bge-reranker-v2-m3")

        indexed_scores = reranker_model.rerank_with_index(
            query=query,
            documents=doc_texts,
            max_length=settings.RERANKER_MAX_LENGTH,
        )

        reranked = []
        for orig_idx, score in indexed_scores:
            doc_copy = documents[orig_idx].copy()
            doc_copy["rerank_score"] = score
            doc_copy["rerank_source"] = "bge-reranker-v2-m3"
            reranked.append(doc_copy)

        logger.info(
            f"✅ Reranked {len(reranked)} chunks. "
            f"Best score: {reranked[0].get('rerank_score', 0):.4f}"
        )
        return reranked

    except Exception as e:
        logger.error(f"❌ Error in document reranking: {e}", exc_info=True)
        return documents


# ============================================================================
# STANDARD SEARCH FUNCTIONS (giữ nguyên từ v3)
# ============================================================================

def pad_vector_to_dimension(vector: np.ndarray, target_dim: int) -> np.ndarray:
    current_dim = vector.shape[0] if vector.ndim == 1 else vector.shape[1]
    if current_dim >= target_dim:
        return vector[:target_dim] if vector.ndim == 1 else vector[:, :target_dim]
    if vector.ndim == 1:
        padding = np.zeros(target_dim - current_dim, dtype=vector.dtype)
        return np.concatenate([vector, padding])
    padding = np.zeros((vector.shape[0], target_dim - current_dim), dtype=vector.dtype)
    return np.concatenate([vector, padding], axis=1)


def safe_encode_and_fix_dimension(
    query: str, target_collection: str, target_field: str
) -> np.ndarray:
    try:
        query_vector = embedding_model.encode_single(query)
        expected_dim = milvus_client._get_collection_dimension(target_collection, target_field)

        if expected_dim > 0 and query_vector.shape[0] != expected_dim:
            logger.warning(
                f"Dimension mismatch. Expected: {expected_dim}, "
                f"Got: {query_vector.shape[0]}. Auto-fixing..."
            )
            query_vector = pad_vector_to_dimension(query_vector, expected_dim)

        return query_vector
    except Exception as e:
        logger.error(f"Error encoding query: {e}")
        raise


@tool
def search_documents(query: str) -> List[Dict[str, Any]]:
    """Tìm kiếm chunk tài liệu liên quan đến câu hỏi (collection document_chunks)."""
    try:
        query_vector = safe_encode_and_fix_dimension(
            query, settings.DOCUMENT_COLLECTION, milvus_client.DOC_VECTOR_FIELD
        )
        return milvus_client.search_documents(query_vector, settings.TOP_K)
    except Exception as e:
        logger.error(f"Error in search_documents: {e}")
        return [{"error": f"Lỗi tìm kiếm tài liệu: {e}"}]


@tool
def search_faq(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """Tìm kiếm FAQ với top_k cao hơn để reranking có nhiều lựa chọn."""
    try:
        if top_k is None:
            top_k = getattr(settings, "FAQ_TOP_K", 10)

        query_vector = safe_encode_and_fix_dimension(
            query, settings.FAQ_COLLECTION, milvus_client.FAQ_VECTOR_FIELD
        )
        results = milvus_client.search_faq(query_vector, top_k)
        logger.info(f"Retrieved {len(results)} FAQ candidates for reranking")
        return results
    except Exception as e:
        logger.error(f"Error in search_faq: {e}")
        return [{"error": f"Lỗi tìm kiếm FAQ: {e}"}]


@tool
def check_database_connection() -> Dict[str, Any]:
    """Kiểm tra kết nối cơ sở dữ liệu và trạng thái reranker."""
    try:
        is_connected = milvus_client.check_connection()
        result = {
            "connected": is_connected,
            "message": "Kết nối bình thường" if is_connected else "Mất kết nối cơ sở dữ liệu",
        }

        if is_connected:
            try:
                test_vector = embedding_model.encode_single("test")
                embedding_dim = test_vector.shape[0]

                doc_dim = milvus_client._get_collection_dimension(
                    settings.DOCUMENT_COLLECTION, milvus_client.DOC_VECTOR_FIELD
                )
                faq_dim = milvus_client._get_collection_dimension(
                    settings.FAQ_COLLECTION, milvus_client.FAQ_VECTOR_FIELD
                )

                result["dimension_info"] = {
                    "embedding_model_dimension": embedding_dim,
                    "document_collection_dimension": doc_dim,
                    "faq_collection_dimension": faq_dim,
                    "dimension_match": {
                        "documents": embedding_dim == doc_dim,
                        "faq": embedding_dim == faq_dim,
                    },
                }
                if embedding_dim != doc_dim or embedding_dim != faq_dim:
                    result["warning"] = (
                        "Dimension mismatch detected — using auto-fix with zero padding"
                    )
            except Exception as dim_error:
                result["dimension_check_error"] = str(dim_error)

        # Trạng thái local reranker thay cho Cohere
        result["local_reranker"] = {
            "available": reranker_model.available,
            "model": settings.RERANKER_MODEL,
            "fp16": settings.RERANKER_USE_FP16,
            "max_length": settings.RERANKER_MAX_LENGTH,
        }

        return result

    except Exception as e:
        return {"connected": False, "message": f"Lỗi kiểm tra kết nối: {e}"}