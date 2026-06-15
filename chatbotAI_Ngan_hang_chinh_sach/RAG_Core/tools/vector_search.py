# RAG_Core/tools/vector_search.py - UPDATED WITH PERSONALIZATION SUPPORT

from langchain_core.tools import tool
from typing import List, Dict, Any
import numpy as np
from models.embedding_model import embedding_model
from database.milvus_client import milvus_client
from config.settings import settings
import logging
import os

logger = logging.getLogger(__name__)

# ============================================================================
# COHERE RERANKER SETUP (STANDALONE - AUTO IMPORT)
# ============================================================================

cohere_client = None
COHERE_RERANK_MODEL = 'rerank-multilingual-v3.0'

try:
    import cohere

    # Tự động lấy API key từ nhiều nguồn (theo thứ tự ưu tiên)
    cohere_api_key = None

    # 1. Thử lấy từ settings (nếu có)
    if hasattr(settings, 'COHERE_API_KEY'):
        cohere_api_key = settings.COHERE_API_KEY
        logger.info("📍 Found COHERE_API_KEY in settings")

    # 2. Thử lấy từ environment variable
    if not cohere_api_key:
        cohere_api_key = os.getenv('COHERE_API_KEY')
        if cohere_api_key:
            logger.info("📍 Found COHERE_API_KEY in environment")

    # 3. Hardcode key (TEMPORARY - chỉ cho dev/testing)
    if not cohere_api_key:
        cohere_api_key = "NoQ9Jjvz5r1JeRWZG8L9dnl8BxYljmnOdiUfTnfk"
        logger.warning("⚠️ Using hardcoded COHERE_API_KEY (not recommended for production)")

    if not cohere_api_key or cohere_api_key == "your-api-key-here":
        raise ValueError("COHERE_API_KEY not configured")

    # Initialize Cohere client
    cohere_client = cohere.Client(cohere_api_key)

    # Lấy model từ settings hoặc dùng default
    if hasattr(settings, 'COHERE_RERANK_MODEL'):
        COHERE_RERANK_MODEL = settings.COHERE_RERANK_MODEL

    logger.info(f"✅ Cohere Reranker initialized with model: {COHERE_RERANK_MODEL}")

    # Test connection
    try:
        test_response = cohere_client.rerank(
            query="test",
            documents=["test document"],
            model=COHERE_RERANK_MODEL,
            top_n=1
        )
        logger.info("✅ Cohere API connection test successful")
    except Exception as test_error:
        logger.warning(f"⚠️ Cohere API test failed: {test_error}")

except ImportError:
    logger.error("❌ Cohere library not installed. Run: pip install cohere")
    cohere_client = None

except Exception as e:
    logger.error(f"❌ Failed to initialize Cohere client: {e}", exc_info=True)
    cohere_client = None


# ============================================================================
# FAQ RERANKING (COHERE API)
# ============================================================================

@tool
def rerank_faq(query: str, faq_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank FAQ sử dụng Cohere Rerank API (tối ưu cho Tiếng Việt)

    UPDATED: Now receives CONTEXTUALIZED question for better accuracy

    Args:
        query: CONTEXTUALIZED question (if follow-up) or original (if standalone)
        faq_results: List of FAQ candidates from vector search

    Returns:
        List of FAQs sorted by rerank_score (descending)
    """
    try:
        logger.info("-" * 50)
        logger.info("🔄 COHERE RERANKER")
        logger.info("-" * 50)
        logger.info(f"📝 Query: '{query[:100]}'")
        logger.info(f"   Length: {len(query)} chars")
        logger.info(f"   FAQs to rerank: {len(faq_results)}")

        if not faq_results:
            logger.warning("⚠️  No FAQ to rerank")
            return []

        if cohere_client is None:
            logger.warning("⚠️  Cohere client not available, returning original FAQs")
            return faq_results

        # Prepare documents cho Cohere
        documents = []
        for i, faq in enumerate(faq_results):
            question = faq.get('question', '').strip()
            answer = faq.get('answer', '').strip()

            # Combine với format rõ ràng
            combined = f"Câu hỏi: {question}\nTrả lời: {answer}"
            documents.append(combined)

            # Log first 2 candidates
            if i < 2:
                logger.info(f"   Candidate {i + 1}: '{question[:60]}...'")

        if not documents:
            logger.warning("⚠️  No valid FAQ documents created")
            return faq_results

        # Call Cohere Rerank API
        logger.info(f"🌐 Calling Cohere API (model: {COHERE_RERANK_MODEL})")

        import time
        start_time = time.time()

        rerank_response = cohere_client.rerank(
            query=query,  # ← ✅ CONTEXTUALIZED query
            documents=documents,
            model=COHERE_RERANK_MODEL,
            top_n=len(documents),
            return_documents=False
        )

        api_time = time.time() - start_time
        logger.info(f"⏱️  Cohere API completed in {api_time:.3f}s")

        # Map scores trở lại FAQs
        reranked_faq = []
        for result in rerank_response.results:
            idx = result.index
            score = result.relevance_score

            faq_copy = faq_results[idx].copy()
            faq_copy['rerank_score'] = float(score)
            faq_copy['rerank_source'] = 'cohere'
            reranked_faq.append(faq_copy)

        # Sort by rerank_score
        reranked_faq.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        # Log top 3 results
        logger.info(f"\n📊 RERANK RESULTS (Top 3):")
        for i, faq in enumerate(reranked_faq[:3], 1):
            logger.info(
                f"   {i}. Score: {faq.get('rerank_score', 0):.3f} | "
                f"Q: '{faq.get('question', '')[:60]}...'"
            )

        logger.info(
            f"\n✅ Reranked {len(reranked_faq)} FAQs successfully\n"
            f"   Best score: {reranked_faq[0].get('rerank_score', 0):.3f}"
        )
        logger.info("-" * 50 + "\n")

        return reranked_faq

    except Exception as e:
        logger.error(f"❌ Error in Cohere FAQ reranking: {e}", exc_info=True)
        logger.info("↩️  Falling back to similarity scores")
        # Fallback to original similarity scores
        return sorted(
            faq_results,
            key=lambda x: x.get('similarity_score', 0),
            reverse=True
        )


# ============================================================================
# DOCUMENT RERANKING (COHERE API)
# ============================================================================

@tool
def rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rerank documents sử dụng Cohere Rerank API
    """
    try:
        if not documents:
            logger.warning("No documents to rerank")
            return []

        if cohere_client is None:
            logger.warning("Cohere client not available, returning original documents")
            return documents

        # Prepare document texts
        doc_texts = []
        for doc in documents:
            doc_text = doc.get('description', '') or doc.get('answer', '') or doc.get('content', '')
            doc_texts.append(doc_text)

        if not doc_texts:
            logger.warning("No valid document texts found")
            return documents

        # Call Cohere Rerank API
        logger.info(f"🔄 Reranking {len(doc_texts)} documents với Cohere API")

        rerank_response = cohere_client.rerank(
            query=query,
            documents=doc_texts,
            model=COHERE_RERANK_MODEL,
            top_n=len(doc_texts),
            return_documents=False
        )

        # Map scores trở lại documents
        reranked_docs = []
        for result in rerank_response.results:
            idx = result.index
            score = result.relevance_score

            doc_copy = documents[idx].copy()
            doc_copy['rerank_score'] = float(score)
            doc_copy['rerank_source'] = 'cohere'
            reranked_docs.append(doc_copy)

        # Sort by rerank_score
        reranked_docs.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        logger.info(
            f"✅ Reranked {len(reranked_docs)} documents. "
            f"Best score: {reranked_docs[0].get('rerank_score', 0):.3f}"
        )

        return reranked_docs

    except Exception as e:
        logger.error(f"Error in Cohere document reranking: {e}", exc_info=True)
        return documents


# ============================================================================
# ADVANCED: HYBRID RERANKING (Optional)
# ============================================================================

@tool
def hybrid_rerank_faq(
        query: str,
        faq_results: List[Dict[str, Any]],
        use_variants: bool = True
) -> List[Dict[str, Any]]:
    """
    Rerank FAQ với multiple strategies để tối ưu kết quả

    Args:
        query: User query
        faq_results: FAQ results from vector search
        use_variants: Nếu True, sẽ test nhiều variants của query
    """
    try:
        if not faq_results or cohere_client is None:
            return rerank_faq(query, faq_results)

        if not use_variants:
            return rerank_faq(query, faq_results)

        # Strategy: Rerank với multiple query variants để có kết quả tốt nhất
        documents = []
        for faq in faq_results:
            question = faq.get('question', '').strip()
            answer = faq.get('answer', '').strip()
            combined = f"Câu hỏi: {question}\nTrả lời: {answer}"
            documents.append(combined)

        # Variant 1: Original query
        logger.info("🔄 Reranking với original query")
        rerank1 = cohere_client.rerank(
            query=query,
            documents=documents,
            model=COHERE_RERANK_MODEL,
            top_n=len(documents),
            return_documents=False
        )

        # Variant 2: Query as a question (nếu chưa phải câu hỏi)
        query_as_question = query if query.strip().endswith('?') else f"{query}?"
        logger.info("🔄 Reranking với question format")
        rerank2 = cohere_client.rerank(
            query=query_as_question,
            documents=documents,
            model=COHERE_RERANK_MODEL,
            top_n=len(documents),
            return_documents=False
        )

        # Combine scores với weighted average
        combined_scores = {}
        weights = [0.6, 0.4]  # Ưu tiên original query hơn

        for result in rerank1.results:
            idx = result.index
            combined_scores[idx] = result.relevance_score * weights[0]

        for result in rerank2.results:
            idx = result.index
            combined_scores[idx] = combined_scores.get(idx, 0) + result.relevance_score * weights[1]

        # Create final ranked list
        reranked_faq = []
        for idx, score in combined_scores.items():
            faq_copy = faq_results[idx].copy()
            faq_copy['rerank_score'] = float(score)
            faq_copy['rerank_source'] = 'cohere_hybrid'
            reranked_faq.append(faq_copy)

        reranked_faq.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        logger.info(
            f"✅ Hybrid reranked {len(reranked_faq)} FAQs. "
            f"Best score: {reranked_faq[0].get('rerank_score', 0):.3f}"
        )

        return reranked_faq

    except Exception as e:
        logger.error(f"Error in hybrid reranking: {e}", exc_info=True)
        return rerank_faq(query, faq_results)


# ============================================================================
# PERSONALIZATION SEARCH FUNCTIONS - NEW
# ============================================================================

@tool
def search_personalization_documents(query: str) -> List[Dict[str, Any]]:
    """Tìm kiếm tài liệu trong personalization database"""
    try:
        from database.personalization_milvus_client import personalization_milvus_client

        # Encode query
        query_vector = embedding_model.encode_single(query)

        # Search in personalization DB
        results = personalization_milvus_client.search_documents(query_vector, settings.TOP_K)
        logger.info(f"✅ Found {len(results)} personalization documents")
        return results

    except Exception as e:
        logger.error(f"Error in search_personalization_documents: {str(e)}")
        return [{"error": f"Lỗi tìm kiếm tài liệu: {str(e)}"}]


@tool
def search_personalization_faq(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Tìm kiếm FAQ trong personalization database
    """
    try:
        from database.personalization_milvus_client import personalization_milvus_client

        if top_k is None:
            top_k = getattr(settings, 'FAQ_TOP_K', 10)

        # Encode query
        query_vector = embedding_model.encode_single(query)

        # Search in personalization FAQ
        results = personalization_milvus_client.search_faq(query_vector, top_k)
        logger.info(f"✅ Retrieved {len(results)} personalization FAQ candidates")

        return results

    except Exception as e:
        logger.error(f"Error in search_personalization_faq: {str(e)}")
        return [{"error": f"Lỗi tìm kiếm FAQ: {str(e)}"}]


# ============================================================================
# STANDARD SEARCH FUNCTIONS (Original DB)
# ============================================================================

def pad_vector_to_dimension(vector: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad vector with zeros to reach target dimension"""
    current_dim = vector.shape[0] if vector.ndim == 1 else vector.shape[1]

    if current_dim >= target_dim:
        return vector[:target_dim] if vector.ndim == 1 else vector[:, :target_dim]

    if vector.ndim == 1:
        padding = np.zeros(target_dim - current_dim, dtype=vector.dtype)
        return np.concatenate([vector, padding])
    else:
        padding = np.zeros((vector.shape[0], target_dim - current_dim), dtype=vector.dtype)
        return np.concatenate([vector, padding], axis=1)


def safe_encode_and_fix_dimension(query: str, target_collection: str, target_field: str) -> np.ndarray:
    """Encode query and automatically fix dimension if needed"""
    try:
        query_vector = embedding_model.encode_single(query)
        expected_dim = milvus_client._get_collection_dimension(target_collection, target_field)

        if expected_dim > 0 and query_vector.shape[0] != expected_dim:
            logger.warning(
                f"Dimension mismatch. Expected: {expected_dim}, Got: {query_vector.shape[0]}. Auto-fixing..."
            )
            query_vector = pad_vector_to_dimension(query_vector, expected_dim)
            logger.info(f"Vector dimension fixed to {expected_dim}")

        return query_vector

    except Exception as e:
        logger.error(f"Error encoding query: {str(e)}")
        raise


@tool
def search_documents(query: str) -> List[Dict[str, Any]]:
    """Tìm kiếm tài liệu liên quan đến câu hỏi (Standard DB)"""
    try:
        query_vector = safe_encode_and_fix_dimension(
            query,
            settings.DOCUMENT_COLLECTION,
            "description_vector"
        )

        results = milvus_client.search_documents(query_vector, settings.TOP_K)
        return results

    except Exception as e:
        logger.error(f"Error in search_documents: {str(e)}")
        return [{"error": f"Lỗi tìm kiếm tài liệu: {str(e)}"}]


@tool
def search_faq(query: str, top_k: int = None) -> List[Dict[str, Any]]:
    """
    Tìm kiếm FAQ với top_k cao hơn để reranking có nhiều lựa chọn (Standard DB)
    """
    try:
        if top_k is None:
            top_k = getattr(settings, 'FAQ_TOP_K', 10)

        query_vector = safe_encode_and_fix_dimension(
            query,
            settings.FAQ_COLLECTION,
            "question_vector"
        )

        results = milvus_client.search_faq(query_vector, top_k)
        logger.info(f"Retrieved {len(results)} FAQ candidates for reranking")

        return results

    except Exception as e:
        logger.error(f"Error in search_faq: {str(e)}")
        return [{"error": f"Lỗi tìm kiếm FAQ: {str(e)}"}]


@tool
def check_database_connection() -> Dict[str, Any]:
    """Kiểm tra kết nối cơ sở dữ liệu"""
    try:
        is_connected = milvus_client.check_connection()

        result = {
            "connected": is_connected,
            "message": "Kết nối bình thường" if is_connected else "Mất kết nối cơ sở dữ liệu"
        }

        if is_connected:
            try:
                test_vector = embedding_model.encode_single("test")
                embedding_dim = test_vector.shape[0]

                doc_dim = milvus_client._get_collection_dimension(
                    settings.DOCUMENT_COLLECTION, "description_vector"
                )
                faq_dim = milvus_client._get_collection_dimension(
                    settings.FAQ_COLLECTION, "question_vector"
                )

                result["dimension_info"] = {
                    "embedding_model_dimension": embedding_dim,
                    "document_collection_dimension": doc_dim,
                    "faq_collection_dimension": faq_dim,
                    "dimension_match": {
                        "documents": embedding_dim == doc_dim,
                        "faq": embedding_dim == faq_dim
                    }
                }

                if embedding_dim != doc_dim or embedding_dim != faq_dim:
                    result["warning"] = "Dimension mismatch detected - using auto-fix with zero padding"

            except Exception as dim_error:
                result["dimension_check_error"] = str(dim_error)

        # Add Cohere status
        result["cohere_reranker"] = {
            "available": cohere_client is not None,
            "model": COHERE_RERANK_MODEL if cohere_client else None
        }

        return result

    except Exception as e:
        return {
            "connected": False,
            "message": f"Lỗi kiểm tra kết nối: {str(e)}"
        }