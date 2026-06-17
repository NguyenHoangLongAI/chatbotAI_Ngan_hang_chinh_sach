# RAG_Core/workflow/rag_workflow.py

from typing import Dict, Any, List, AsyncIterator
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from agents.supervisor import SupervisorAgent
from agents.faq_agent import FAQAgent
from agents.retriever_agent import RetrieverAgent
from agents.grader_agent import GraderAgent
from agents.generator_agent import GeneratorAgent
from agents.reporter_agent import ReporterAgent
from agents.hello_agent import HelloAgent  # ← NEW

from agents.base_agent import (
    StreamingChatterAgent,
    StreamingOtherAgent,
    StreamingNotEnoughInfoAgent
)

from services.document_url_service import document_url_service

logger = logging.getLogger(__name__)


class ChatbotState(TypedDict):
    question: str
    original_question: str
    history: List[Dict[str, str]]
    is_followup: bool
    context_summary: str
    relevant_context: str
    current_agent: str
    documents: List[Dict[str, Any]]
    qualified_documents: List[Dict[str, Any]]
    references: List[Dict[str, Any]]
    answer: str
    status: str
    iteration_count: int
    supervisor_classification: Dict[str, Any]
    faq_result: Dict[str, Any]
    retriever_result: Dict[str, Any]
    parallel_mode: bool
    streaming_mode: bool


class RAGWorkflow:
    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.faq_agent = FAQAgent()
        self.retriever_agent = RetrieverAgent()
        self.grader_agent = GraderAgent()
        self.generator_agent = GeneratorAgent()
        self.reporter_agent = ReporterAgent()
        self.hello_agent = HelloAgent()  # ← NEW

        self.chatter_agent = StreamingChatterAgent()
        self.other_agent = StreamingOtherAgent()
        self.not_enough_info_agent = StreamingNotEnoughInfoAgent()

        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="RAG-Worker")
        self.workflow = self._create_workflow()

    def _create_workflow(self):
        workflow = StateGraph(ChatbotState)

        workflow.add_node("parallel_execution", self._parallel_execution_node)
        workflow.add_node("decision_router", self._decision_router_node)
        workflow.add_node("grader", self._grader_node)
        workflow.add_node("generator", self._generator_node)
        workflow.add_node("not_enough_info", self._not_enough_info_node)
        workflow.add_node("chatter", self._chatter_node)
        workflow.add_node("hello", self._hello_node)          # ← NEW
        workflow.add_node("reporter", self._reporter_node)
        workflow.add_node("other", self._other_node)

        workflow.set_entry_point("parallel_execution")
        workflow.add_edge("parallel_execution", "decision_router")

        workflow.add_conditional_edges(
            "decision_router", self._route_after_decision,
            {
                "GRADER": "grader",
                "CHATTER": "chatter",
                "HELLO": "hello",               # ← NEW
                "REPORTER": "reporter",
                "OTHER": "other",
                "end": "__end__"
            }
        )
        workflow.add_conditional_edges(
            "grader", self._route_next_agent,
            {"GENERATOR": "generator", "NOT_ENOUGH_INFO": "not_enough_info"}
        )

        workflow.add_edge("generator", "__end__")
        workflow.add_edge("not_enough_info", "__end__")
        workflow.add_edge("chatter", "__end__")
        workflow.add_edge("hello", "__end__")   # ← NEW
        workflow.add_edge("reporter", "__end__")
        workflow.add_edge("other", "__end__")

        return workflow.compile()

    # ------------------------------------------------------------------ #
    # HELLO NODE (non-streaming)                                           #
    # ------------------------------------------------------------------ #

    def _hello_node(self, state: ChatbotState) -> ChatbotState:
        try:
            result = self.hello_agent.process(state["question"])
            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state
        except Exception as e:
            logger.error(f"Hello node error: {e}")
            state["answer"] = "Xin chào! Tôi có thể giúp gì cho bạn?"
            return state

    # ------------------------------------------------------------------ #
    # Giữ nguyên toàn bộ các node/method khác từ bản cũ                   #
    # Chỉ cập nhật _decision_router_node và run_with_streaming             #
    # ------------------------------------------------------------------ #

    def _enrich_references_with_urls(self, references):
        try:
            if not references:
                return []
            enriched = document_url_service.enrich_references_with_urls(references)
            urls_added = sum(1 for ref in enriched if ref.get('url'))
            if urls_added > 0:
                logger.info(f"🔗 Enriched {urls_added}/{len(references)} items with URLs")
            return enriched
        except Exception as e:
            logger.error(f"Error enriching references: {e}")
            return references

    def _parallel_execution_node(self, state: ChatbotState) -> ChatbotState:
        question = state["question"]
        history = state.get("history", [])
        skip_faq = state.get("streaming_mode", False)

        logger.info("🚀 Starting parallel execution")
        if skip_faq:
            logger.info("⏭️  Skipping FAQ non-streaming (streaming mode active)")

        supervisor_result = self._get_result_with_timeout(
            self.executor.submit(self._safe_execute_supervisor, question, history),
            timeout=20,
            default={"agent": "FAQ", "contextualized_question": question, "is_followup": False, "context_summary": ""},
            name="Supervisor"
        )

        context_summary = supervisor_result.get("context_summary", "")
        is_followup = supervisor_result.get("is_followup", False)
        contextualized_question = supervisor_result.get("contextualized_question", question)

        logger.info(
            f"📋 Supervisor: agent={supervisor_result.get('agent')}, follow-up={is_followup}\n"
            f"   Original: {question[:60]}\n"
            f"   Contextualized: {contextualized_question[:60]}"
        )

        # HELLO agent → skip FAQ + RETRIEVER hoàn toàn
        supervisor_agent = supervisor_result.get("agent", "FAQ")
        if supervisor_agent == "HELLO":
            faq_result = {"status": "SKIPPED", "answer": "", "references": []}
            retriever_result = {"status": "SKIPPED", "documents": []}
        elif skip_faq:
            faq_result = {"status": "SKIPPED", "answer": "", "references": [], "message": "FAQ skipped for streaming mode"}
            future_retriever = self.executor.submit(
                self._safe_execute_retriever, question, contextualized_question, is_followup
            )
            retriever_result = self._get_result_with_timeout(
                future_retriever, timeout=10,
                default={"status": "ERROR", "documents": []}, name="RETRIEVER"
            )
        else:
            future_faq = self.executor.submit(
                self._safe_execute_faq, contextualized_question, is_followup, context_summary
            )
            faq_result = self._get_result_with_timeout(
                future_faq, timeout=10,
                default={"status": "ERROR", "answer": "", "references": []}, name="FAQ"
            )
            if faq_result.get("references"):
                faq_result["references"] = self._enrich_references_with_urls(faq_result["references"])

            future_retriever = self.executor.submit(
                self._safe_execute_retriever, question, contextualized_question, is_followup
            )
            retriever_result = self._get_result_with_timeout(
                future_retriever, timeout=10,
                default={"status": "ERROR", "documents": []}, name="RETRIEVER"
            )

        state["supervisor_classification"] = supervisor_result
        state["question"] = contextualized_question
        state["original_question"] = question
        state["is_followup"] = is_followup
        state["context_summary"] = context_summary
        state["faq_result"] = faq_result
        state["retriever_result"] = retriever_result
        state["parallel_mode"] = True

        logger.info(
            f"✅ Parallel execution completed:\n"
            f"  - FAQ: {faq_result.get('status')}\n"
            f"  - RETRIEVER: {retriever_result.get('status')}"
        )
        return state

    def _decision_router_node(self, state: ChatbotState) -> ChatbotState:
        supervisor_agent = state.get("supervisor_classification", {}).get("agent", "FAQ")
        faq_result = state.get("faq_result", {})
        retriever_result = state.get("retriever_result", {})

        logger.info(f"🤔 Decision Router: Supervisor={supervisor_agent}")

        # ← HELLO: route thẳng, không qua RAG
        if supervisor_agent == "HELLO":
            state["current_agent"] = "HELLO"
            return state

        if supervisor_agent in ["CHATTER", "REPORTER", "OTHER"]:
            state["current_agent"] = supervisor_agent
            return state

        if faq_result.get("status") == "SUCCESS":
            logger.info("→ FAQ has answer")
            state["status"] = faq_result["status"]
            state["answer"] = faq_result.get("answer", "")
            state["references"] = faq_result.get("references", [])
            state["current_agent"] = "end"
            return state

        if retriever_result.get("documents"):
            logger.info("→ RETRIEVER → GRADER")
            state["documents"] = retriever_result.get("documents", [])
            state["status"] = retriever_result.get("status", "SUCCESS")
            state["current_agent"] = "GRADER"
            return state

        state["current_agent"] = "NOT_ENOUGH_INFO"
        return state

    def _get_result_with_timeout(self, future, timeout, default, name):
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            logger.warning(f"⏱️ {name} timeout, using fallback")
            return default
        except Exception as e:
            logger.error(f"❌ {name} error: {e}")
            return default

    def _safe_execute_supervisor(self, question, history):
        try:
            return self.supervisor.classify_request(question, history)
        except Exception as e:
            logger.error(f"❌ Supervisor error: {e}")
            return {"agent": "FAQ", "contextualized_question": question, "context_summary": "", "is_followup": False}

    def _safe_execute_faq(self, question, is_followup=False, context_summary=""):
        try:
            return self.faq_agent.process(question=question, is_followup=is_followup, context=context_summary)
        except Exception as e:
            logger.error(f"FAQ error: {e}")
            return {"status": "ERROR", "answer": "", "references": [], "next_agent": "RETRIEVER"}

    def _safe_execute_retriever(self, original_question, contextualized_question, is_followup=False):
        try:
            return self.retriever_agent.process(
                question=original_question,
                contextualized_question=contextualized_question,
                is_followup=is_followup
            )
        except Exception as e:
            logger.error(f"RETRIEVER error: {e}")
            return {"status": "ERROR", "documents": [], "next_agent": "NOT_ENOUGH_INFO"}

    def _grader_node(self, state):
        try:
            question = state["question"]
            original_question = state.get("original_question", question)
            documents = state.get("documents", [])
            is_followup = state.get("is_followup", False)

            logger.info(f"📊 Grader: Processing {len(documents)} documents")

            import inspect
            sig = inspect.signature(self.grader_agent.process)
            if 'contextualized_question' in sig.parameters:
                result = self.grader_agent.process(
                    question=original_question, documents=documents,
                    contextualized_question=question, is_followup=is_followup
                )
            else:
                result = self.grader_agent.process(question=question, documents=documents)

            if result.get("references"):
                result["references"] = self._enrich_references_with_urls(result["references"])

            state["status"] = result["status"]
            state["qualified_documents"] = result.get("qualified_documents", [])
            state["references"] = result.get("references", [])

            url_map = {
                ref.get("document_id"): {"url": ref.get("url", ""), "filename": ref.get("filename", "")}
                for ref in state.get("references", []) if ref.get("url")
            }
            if url_map:
                updated_docs = []
                for doc in state.get("qualified_documents", []):
                    doc_copy = doc.copy()
                    doc_id = doc_copy.get("document_id", "")
                    if doc_id in url_map:
                        doc_copy["url"] = url_map[doc_id]["url"]
                        doc_copy["filename"] = url_map[doc_id]["filename"]
                    updated_docs.append(doc_copy)
                state["qualified_documents"] = updated_docs

            state["current_agent"] = result.get("next_agent", "GENERATOR")
            logger.info(f"📊 Grader result: {result['status']}")
            return state

        except Exception as e:
            logger.error(f"❌ Grader node error: {e}", exc_info=True)
            state["current_agent"] = "NOT_ENOUGH_INFO"
            return state

    def _generator_node(self, state):
        try:
            result = self.generator_agent.process(
                question=state["question"],
                documents=state.get("qualified_documents", []),
                references=state.get("references", []),
                history=state.get("history", []),
                is_followup=state.get("is_followup", False),
                context_summary=state.get("context_summary", "")
            )
            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state
        except Exception as e:
            logger.error(f"Generator error: {e}")
            state["answer"] = "Lỗi tạo câu trả lời"
            state["current_agent"] = "end"
            return state

    def _not_enough_info_node(self, state):
        try:
            result = self.not_enough_info_agent.process(
                state["question"], is_followup=state.get("is_followup", False)
            )
            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state
        except Exception as e:
            logger.error(f"Not enough info error: {e}")
            state["answer"] = "Không tìm thấy thông tin"
            return state

    def _chatter_node(self, state):
        try:
            result = self.chatter_agent.process(state["question"], state.get("history", []))
            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state
        except Exception as e:
            logger.error(f"Chatter error: {e}")
            state["answer"] = "Tôi hiểu cảm xúc của bạn"
            return state

    def _reporter_node(self, state):
        try:
            result = self.reporter_agent.process(state["question"])
            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state
        except Exception as e:
            logger.error(f"Reporter error: {e}")
            state["answer"] = "Hệ thống đang bảo trì"
            return state

    def _other_node(self, state):
        try:
            result = self.other_agent.process(state["question"])
            state["status"] = result["status"]
            state["answer"] = result.get("answer", "")
            state["references"] = result.get("references", [])
            state["current_agent"] = "end"
            return state
        except Exception as e:
            logger.error(f"Other error: {e}")
            state["answer"] = "Đây không phải tác vụ của tôi"
            return state

    def _route_after_decision(self, state):
        return state.get("current_agent", "end")

    def _route_next_agent(self, state):
        return state.get("current_agent", "end")

    # ------------------------------------------------------------------ #
    # NON-STREAMING RUN                                                    #
    # ------------------------------------------------------------------ #

    def run(self, question: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        try:
            initial_state = self._create_initial_state(question, history, streaming_mode=False)
            logger.info(f"🚀 Workflow start (non-streaming): {question[:100]}")
            final_state = self.workflow.invoke(initial_state)
            return {
                "answer": final_state.get("answer", "Lỗi xử lý"),
                "references": final_state.get("references", []),
                "status": final_state.get("status", "ERROR")
            }
        except Exception as e:
            logger.error(f"❌ Workflow error: {e}", exc_info=True)
            return {"answer": "Xin lỗi, hệ thống gặp sự cố.", "references": [], "status": "ERROR"}

    # ------------------------------------------------------------------ #
    # STREAMING RUN                                                        #
    # ------------------------------------------------------------------ #

    async def run_with_streaming(self, question: str, history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        try:
            logger.info(f"🚀 Streaming workflow start: {question[:100]}")
            initial_state = self._create_initial_state(question, history, streaming_mode=True)

            state = self._parallel_execution_node(initial_state)
            state = self._decision_router_node(state)

            current_agent = state.get("current_agent")
            supervisor_agent = state.get("supervisor_classification", {}).get("agent")

            logger.info(f"📍 Routed to: {current_agent} (supervisor: {supervisor_agent})")

            # ── HELLO ──────────────────────────────────────────────────
            if supervisor_agent == "HELLO":
                logger.info("👋 STREAMING: HELLO")
                return {
                    "answer_stream": self.hello_agent.process_streaming(question=state["question"]),
                    "references": [],
                    "status": "STREAMING"
                }

            # ── FAQ ────────────────────────────────────────────────────
            if supervisor_agent == "FAQ":
                logger.info("🔍 FAQ classified - checking confidence...")
                from tools.vector_search import search_faq, rerank_faq
                from config.settings import settings

                faq_results = search_faq.invoke({"query": state["question"]})

                if not faq_results or "error" in str(faq_results):
                    logger.warning("❌ FAQ vector search failed → GRADER")
                    current_agent = "GRADER"
                else:
                    filtered_faqs = [
                        faq for faq in faq_results
                        if faq.get("similarity_score", 0) >= settings.FAQ_VECTOR_THRESHOLD
                    ]

                    if not filtered_faqs:
                        logger.info("⚠️  No FAQ above vector threshold → GRADER")
                        current_agent = "GRADER"
                    else:
                        reranked_faqs = rerank_faq.invoke({"query": state["question"], "faq_results": filtered_faqs})

                        if not reranked_faqs:
                            logger.warning("❌ FAQ rerank failed → GRADER")
                            current_agent = "GRADER"
                        else:
                            best_score = reranked_faqs[0].get("rerank_score", 0)
                            logger.info(f"📊 FAQ best rerank score: {best_score:.3f}")

                            if best_score >= settings.FAQ_RERANK_THRESHOLD:
                                logger.info("✅ FAQ CONFIDENT - Streaming answer")
                                return {
                                    "answer_stream": self.faq_agent.process_streaming(
                                        question=state["question"],
                                        reranked_faqs=reranked_faqs,
                                        is_followup=state.get("is_followup", False),
                                        context=state.get("context_summary", "")
                                    ),
                                    "references": [{
                                        "document_id": reranked_faqs[0].get("faq_id"),
                                        "type": "FAQ",
                                        "description": reranked_faqs[0].get("question", ""),
                                        "rerank_score": round(best_score, 4)
                                    }],
                                    "status": "STREAMING"
                                }
                            else:
                                logger.info(f"⚠️  FAQ not confident ({best_score:.3f} < {settings.FAQ_RERANK_THRESHOLD}) → GRADER")
                                current_agent = "GRADER"

            # ── GRADER → GENERATOR / NOT_ENOUGH_INFO ───────────────────
            if current_agent == "GRADER":
                state = self._grader_node(state)

                if state.get("current_agent") == "GENERATOR":
                    logger.info("✅ STREAMING: GENERATOR")
                    return {
                        "answer_stream": self.generator_agent.process_streaming(
                            question=state["question"],
                            documents=state.get("qualified_documents", []),
                            references=state.get("references", []),
                            history=history or [],
                            is_followup=state.get("is_followup", False),
                            context_summary=state.get("context_summary", "")
                        ),
                        "references": state.get("references", []),
                        "status": "STREAMING"
                    }
                else:
                    logger.info("✅ STREAMING: NOT_ENOUGH_INFO")
                    from config.settings import settings
                    return {
                        "answer_stream": self.not_enough_info_agent.process_streaming(
                            question=state["question"],
                            support_phone=settings.SUPPORT_PHONE
                        ),
                        "references": [],   # đã sửa: bỏ {"document_id": "llm_knowledge", "type": "GENERAL_KNOWLEDGE"}
                        "status": "STREAMING"
                    }

            # ── CHATTER ────────────────────────────────────────────────
            elif current_agent == "CHATTER":
                logger.info("✅ STREAMING: CHATTER")
                from config.settings import settings
                return {
                    "answer_stream": self.chatter_agent.process_streaming(
                        question=state["question"],
                        history=state.get("history", []),
                        support_phone=settings.SUPPORT_PHONE
                    ),
                    "references": [{"document_id": "support_contact", "type": "SUPPORT"}],
                    "status": "STREAMING"
                }

            # ── OTHER ──────────────────────────────────────────────────
            elif current_agent == "OTHER":
                logger.info("✅ STREAMING: OTHER")
                from config.settings import settings
                return {
                    "answer_stream": self.other_agent.process_streaming(
                        question=state["question"],
                        support_phone=settings.SUPPORT_PHONE
                    ),
                    "references": [],
                    "status": "STREAMING"
                }

            # ── REPORTER ───────────────────────────────────────────────
            elif current_agent == "REPORTER":
                logger.info("📋 REPORTER")
                state = self._reporter_node(state)
                answer_text = state.get("answer", "")

                async def reporter_generator():
                    for word in answer_text.split():
                        yield word + " "
                        await asyncio.sleep(0.01)

                return {
                    "answer_stream": reporter_generator(),
                    "references": state.get("references", []),
                    "status": state.get("status", "SUCCESS")
                }

            else:
                logger.warning(f"Unknown agent: {current_agent}")

                async def error_generator():
                    yield "Xin lỗi, không thể xử lý yêu cầu này."

                return {"answer_stream": error_generator(), "references": [], "status": "ERROR"}

        except Exception as e:
            logger.error(f"❌ Streaming workflow error: {e}", exc_info=True)

            async def error_generator():
                yield "Xin lỗi, hệ thống gặp sự cố."

            return {"answer_stream": error_generator(), "references": [], "status": "ERROR"}

    def _create_initial_state(self, question, history=None, streaming_mode=False) -> ChatbotState:
        return ChatbotState(
            question=question, original_question=question,
            history=history or [], is_followup=False,
            context_summary="", relevant_context="",
            current_agent="parallel_execution",
            documents=[], qualified_documents=[],
            references=[], answer="", status="",
            iteration_count=0, supervisor_classification={},
            faq_result={}, retriever_result={},
            parallel_mode=False, streaming_mode=streaming_mode
        )

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True, timeout=5)