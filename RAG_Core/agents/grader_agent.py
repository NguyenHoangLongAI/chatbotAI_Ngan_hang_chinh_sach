# RAG_Core/agents/grader_agent.py

from typing import Dict, Any, List
from tools.vector_search import rerank_documents
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class GraderAgent:
    def __init__(self):
        self.name = "GRADER"
        # NEW: dùng đúng threshold trong settings (trước đây hardcode 0.6)
        self.reranking_threshold = settings.DOCUMENT_RERANK_THRESHOLD

    def process(
            self,
            question: str,
            documents: List[Dict[str, Any]],
            contextualized_question: str = "",
            is_followup: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        try:
            if not documents:
                return {
                    "status": "INSUFFICIENT",
                    "qualified_documents": [],
                    "references": [],
                    "next_agent": "NOT_ENOUGH_INFO"
                }

            if is_followup and contextualized_question:
                rerank_query = contextualized_question
                logger.info("📝 Using CONTEXTUALIZED QUESTION for reranking (follow-up)")
            else:
                rerank_query = question
                logger.info("📝 Using ORIGINAL QUESTION for reranking")

            logger.info(f"🔄 Reranking {len(documents)} chunks")
            reranked_docs = rerank_documents.invoke({"query": rerank_query, "documents": documents})

            if not reranked_docs:
                raise RuntimeError("Reranking failed: empty results")

            qualified_docs = [
                doc for doc in reranked_docs
                if doc.get("rerank_score", 0) >= self.reranking_threshold
            ]

            if qualified_docs:
                logger.info(f"✅ Found {len(qualified_docs)} qualified chunks")
                return {
                    "status": "SUFFICIENT",
                    "qualified_documents": qualified_docs,
                    "references": [
                        {
                            "document_id": doc.get("document_id"),
                            "type": "DOCUMENT",
                            "description": (doc.get("content") or "")[:300],
                            "section_path": doc.get("section_path", ""),
                            "page_num": doc.get("page_num", 0),
                            "chunk_type": doc.get("chunk_type", ""),
                            "rerank_score": round(doc.get("rerank_score", 0), 5),
                            "similarity_score": round(doc.get("similarity_score", 0), 5),
                            "reranked_with": "contextualized_question" if (is_followup and contextualized_question) else "original_question"
                        }
                        for doc in qualified_docs
                    ],
                    "next_agent": "GENERATOR"
                }

            logger.warning("No chunks passed grading thresholds")
            return {
                "status": "INSUFFICIENT",
                "qualified_documents": [],
                "references": [],
                "next_agent": "NOT_ENOUGH_INFO"
            }

        except RuntimeError as e:
            logger.error(f"❌ Critical error in grader agent: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error in grader agent: {e}", exc_info=True)
            raise RuntimeError(f"Grader agent failed: {e}") from e