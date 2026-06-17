# RAG_Core/agents/retriever_agent.py

from typing import Dict, Any, List
from models.llm_model import llm_model
from tools.vector_search import search_documents
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class RetrieverAgent:
    """Tìm chunk tài liệu liên quan từ collection document_chunks."""

    def __init__(self):
        self.name = "RETRIEVER"
        self.tools = [search_documents]

    def process(
            self,
            question: str,
            contextualized_question: str = "",
            is_followup: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        try:
            if is_followup or contextualized_question:
                search_query = contextualized_question
                logger.info("🔍 Using CONTEXTUALIZED QUESTION for vector search (follow-up)")
            else:
                search_query = question
                logger.info("🔍 Using ORIGINAL QUESTION for vector search")

            logger.info(f"📚 Searching chunks with query: {search_query[:100]}...")
            search_results = search_documents.invoke({"query": search_query})

            if not search_results or "error" in str(search_results):
                logger.warning("Vector search failed or returned error")
                return {"status": "ERROR", "documents": [], "next_agent": "NOT_ENOUGH_INFO"}

            relevant_docs = [
                doc for doc in search_results
                if doc.get("similarity_score", 0) > settings.SIMILARITY_THRESHOLD
            ]

            if not relevant_docs:
                logger.info(
                    f"No chunks above threshold {settings.SIMILARITY_THRESHOLD}, "
                    f"returning all {len(search_results)} for grader"
                )
                return {
                    "status": "NOT_FOUND",
                    "documents": search_results,
                    "search_query_used": "contextualized" if (is_followup and contextualized_question) else "original",
                    "next_agent": "GRADER"
                }

            logger.info(f"✅ Found {len(relevant_docs)} relevant chunks")
            return {
                "status": "SUCCESS",
                "documents": relevant_docs,
                "search_query_used": "contextualized" if (is_followup and contextualized_question) else "original",
                "next_agent": "GRADER"
            }

        except Exception as e:
            logger.error(f"❌ Retriever error: {e}", exc_info=True)
            return {"status": "ERROR", "documents": [], "next_agent": "REPORTER"}