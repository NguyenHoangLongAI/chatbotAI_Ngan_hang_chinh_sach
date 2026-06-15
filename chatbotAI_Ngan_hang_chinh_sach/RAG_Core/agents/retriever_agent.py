# RAG_Core/agents/retriever_agent.py - FIXED: Use contextualized_question

from typing import Dict, Any, List
from models.llm_model import llm_model
from tools.vector_search import search_documents
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RetrieverAgent:
    def __init__(self):
        self.name = "RETRIEVER"
        self.tools = [search_documents]

    def process(
            self,
            question: str,
            contextualized_question: str = "",  # NEW: Accept contextualized question
            is_followup: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Tìm kiếm tài liệu liên quan đến câu hỏi

        Args:
            question: Câu hỏi gốc (for logging)
            contextualized_question: Câu hỏi đã được làm rõ (dùng để search)
            is_followup: Có phải follow-up question không
        """
        try:
            # ================================================================
            # QUYẾT ĐỊNH QUERY CHO VECTOR SEARCH
            # ================================================================

            # FIXED: Nếu là follow-up và có contextualized_question → dùng nó
            if is_followup or contextualized_question:
                search_query = contextualized_question
                logger.info(f"🔍 Using CONTEXTUALIZED QUESTION for vector search (follow-up)")
                logger.debug(f"Original: {question[:60]}")
                logger.debug(f"Contextualized: {contextualized_question[:100]}")
            else:
                # Không phải follow-up hoặc không có contextualized → dùng câu hỏi gốc
                search_query = question
                logger.info(f"🔍 Using ORIGINAL QUESTION for vector search")

            # ================================================================
            # VECTOR SEARCH
            # ================================================================

            logger.info(f"📚 Searching documents with query: {search_query[:100]}...")

            # Tìm kiếm tài liệu với query đã quyết định
            search_results = search_documents.invoke({"query": search_query})

            if not search_results or "error" in str(search_results):
                logger.warning("Vector search failed or returned error")
                return {
                    "status": "ERROR",
                    "documents": [],
                    "next_agent": "NOT_ENOUGH_INFO"
                }

            # ================================================================
            # LỌC KẾT QUẢ THEO SIMILARITY THRESHOLD
            # ================================================================

            relevant_docs = [
                doc for doc in search_results
                if doc.get("similarity_score", 0) > settings.SIMILARITY_THRESHOLD
            ]

            if not relevant_docs:
                logger.info(
                    f"No documents above threshold {settings.SIMILARITY_THRESHOLD}, "
                    f"returning all {len(search_results)} for grader"
                )
                return {
                    "status": "NOT_FOUND",
                    "documents": search_results,  # Pass all to GRADER for reranking
                    "search_query_used": "contextualized" if (is_followup and contextualized_question) else "original",
                    "next_agent": "GRADER"
                }

            logger.info(
                f"✅ Found {len(relevant_docs)} relevant documents "
                f"(searched with {'contextualized question' if is_followup and contextualized_question else 'original question'})"
            )

            return {
                "status": "SUCCESS",
                "documents": relevant_docs,
                "search_query_used": "contextualized" if (is_followup and contextualized_question) else "original",
                "next_agent": "GRADER"
            }

        except Exception as e:
            logger.error(f"❌ Retriever error: {e}", exc_info=True)
            return {
                "status": "ERROR",
                "documents": [],
                "next_agent": "REPORTER"
            }