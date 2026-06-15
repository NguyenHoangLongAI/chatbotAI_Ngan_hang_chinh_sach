# RAG_Core/agents/grader_agent.py - FIXED: Use contextualized_question

from typing import Dict, Any, List
from tools.vector_search import rerank_documents
from config.settings import settings
import logging

logger = logging.getLogger(__name__)


class GraderAgent:
    def __init__(self):
        self.name = "GRADER"
        self.reranking_threshold = 0.6

    def process(
            self,
            question: str,
            documents: List[Dict[str, Any]],
            contextualized_question: str = "",  # NEW: Accept contextualized question
            is_followup: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Đánh giá chất lượng tài liệu bằng reranking model

        Args:
            question: Câu hỏi gốc (for logging)
            documents: Danh sách tài liệu
            contextualized_question: Câu hỏi đã được làm rõ (dùng cho rerank)
            is_followup: Có phải follow-up question không
        """
        try:
            if not documents:
                logger.warning("No documents to grade")
                return {
                    "status": "INSUFFICIENT",
                    "qualified_documents": [],
                    "references": [],
                    "next_agent": "NOT_ENOUGH_INFO"
                }

            # ================================================================
            # QUYẾT ĐỊNH QUERY CHO RERANKING
            # ================================================================

            # FIXED: Nếu là follow-up và có contextualized_question → dùng nó
            if is_followup and contextualized_question:
                rerank_query = contextualized_question
                logger.info(f"📝 Using CONTEXTUALIZED QUESTION for reranking (follow-up)")
                logger.debug(f"Original: {question[:60]}")
                logger.debug(f"Contextualized: {contextualized_question[:100]}")
            else:
                # Không phải follow-up hoặc không có contextualized → dùng câu hỏi gốc
                rerank_query = question
                logger.info(f"📝 Using ORIGINAL QUESTION for reranking")

            # ================================================================
            # RERANK DOCUMENTS
            # ================================================================

            logger.info(f"🔄 Reranking {len(documents)} documents")
            logger.debug(f"Rerank query: {rerank_query[:100]}...")

            reranked_docs = rerank_documents.invoke({
                "query": rerank_query,  # ✅ Sử dụng contextualized question
                "documents": documents
            })

            if not reranked_docs:
                logger.error("❌ Reranking returned empty results")
                raise RuntimeError("Reranking failed: empty results")

            # ================================================================
            # LỌC DOCUMENTS THEO THRESHOLD
            # ================================================================

            qualified_docs = []
            for doc in reranked_docs:
                rerank_score = doc.get("rerank_score", 0)

                if rerank_score >= self.reranking_threshold:
                    qualified_docs.append(doc)
                    logger.debug(
                        f"✓ Doc {doc.get('document_id')}: "
                        f"rerank={rerank_score:.3f}"
                    )
                else:
                    logger.debug(
                        f"✗ Doc {doc.get('document_id')}: "
                        f"rerank={rerank_score:.3f} (below threshold)"
                    )

            # ================================================================
            # QUYẾT ĐỊNH KẾT QUẢ
            # ================================================================

            if qualified_docs:
                logger.info(
                    f"✅ Found {len(qualified_docs)} qualified documents "
                    f"(reranked with {'contextualized question' if is_followup and contextualized_question else 'original question'})"
                )

                return {
                    "status": "SUFFICIENT",
                    "qualified_documents": qualified_docs,
                    "references": [
                        {
                            "document_id": doc.get("document_id"),
                            "type": "DOCUMENT",
                            "description": doc.get("description", ""),
                            "rerank_score": round(doc.get("rerank_score", 0), 5),
                            "similarity_score": round(doc.get("similarity_score", 0), 5),
                            "reranked_with": "contextualized_question" if (is_followup and contextualized_question) else "original_question"
                        }
                        for doc in qualified_docs
                    ],
                    "next_agent": "GENERATOR"
                }
            else:
                logger.warning("No documents passed grading thresholds")
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