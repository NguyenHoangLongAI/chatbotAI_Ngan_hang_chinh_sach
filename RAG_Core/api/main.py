# RAG_Core/api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging
from typing import List, AsyncIterator
import json
import asyncio

from .schemas import (
    ChatRequest, ChatResponse, StreamChunk,
    HealthResponse, DocumentReference
)
from workflow.rag_workflow import RAGWorkflow
from database.milvus_client import milvus_client
from services.document_url_service import document_url_service

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VBSP Internal RAG Chatbot API",
    description="API trợ lý ảo nội bộ VBSP — RAG multi-agent với streaming và trích dẫn URL tài liệu",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_workflow = None


@app.on_event("startup")
async def startup_event():
    global rag_workflow
    try:
        rag_workflow = RAGWorkflow()
        logger.info("✅ RAG Workflow initialized successfully")
        logger.info("✅ Document URL service initialized")
    except Exception as e:
        logger.error(f"⚠️  Failed to initialize RAG Workflow: {e}")


@app.get("/", response_model=dict)
async def root():
    from config.settings import settings
    return {
        "service": "VBSP Internal RAG Chatbot API",
        "version": "3.0.0",
        "assistant": settings.ASSISTANT_NAME,
        "bank": settings.BANK_FULL_NAME,
        "features": ["streaming", "multi-agent", "context-aware", "document-urls", "inline-citations"],
        "endpoints": {"chat": "/chat", "health": "/health"},
        "url_config": {
            "ngrok_enabled": settings.NGROK_PUBLIC_URL is not None,
            "url_replacement_enabled": settings.ENABLE_URL_REPLACEMENT
        }
    }


def enrich_references_with_urls(references: List[dict]) -> List[dict]:
    try:
        return document_url_service.enrich_references_with_urls(references)
    except Exception as e:
        logger.error(f"Error enriching references: {e}")
        return references


async def generate_streaming_response(question: str, history: List) -> AsyncIterator[str]:
    try:
        logger.info(f"🚀 Starting streaming for: {question[:50]}...")

        start_chunk = {"type": "start", "content": None, "references": None, "status": "processing"}
        yield f"data: {json.dumps(start_chunk)}\n\n"
        await asyncio.sleep(0.01)

        result = await rag_workflow.run_with_streaming(question, history)

        answer_stream = result.get("answer_stream")
        references = result.get("references", [])

        if answer_stream:
            chunk_count = 0
            async for chunk in answer_stream:
                if chunk:
                    chunk_count += 1
                    chunk_data = {"type": "chunk", "content": chunk, "references": None, "status": None}
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                    await asyncio.sleep(0.001)
            logger.info(f"✅ Streamed {chunk_count} chunks")
        else:
            error_chunk = {"type": "chunk", "content": "Không thể tạo câu trả lời.", "references": None, "status": None}
            yield f"data: {json.dumps(error_chunk)}\n\n"

        if references:
            enriched_refs = enrich_references_with_urls(references)
            serializable_refs = []
            for ref in enriched_refs:
                ref_dict = {
                    "document_id": ref.get("document_id", ""),
                    "type": ref.get("type", "DOCUMENT"),
                    "description": ref.get("description", ""),
                    "section_path": ref.get("section_path", ""),
                    "page_num": ref.get("page_num", 0),
                }
                if ref.get("url"):
                    ref_dict["url"] = ref["url"]
                    ref_dict["filename"] = ref.get("filename", "")
                    ref_dict["file_type"] = ref.get("file_type", "")
                serializable_refs.append(ref_dict)

            ref_chunk = {"type": "references", "content": None, "references": serializable_refs, "status": None}
            yield f"data: {json.dumps(ref_chunk)}\n\n"
            logger.info(f"📚 Sent {len(serializable_refs)} enriched references")

        end_chunk = {"type": "end", "content": None, "references": None, "status": result.get("status", "SUCCESS")}
        yield f"data: {json.dumps(end_chunk)}\n\n"

    except Exception as e:
        logger.error(f"❌ Streaming error: {e}", exc_info=True)
        error_chunk = {"type": "error", "content": f"Lỗi: {str(e)}", "references": None, "status": "ERROR"}
        yield f"data: {json.dumps(error_chunk)}\n\n"


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        if not rag_workflow:
            raise HTTPException(503, "Workflow not initialized")

        logger.info(f"📨 Question: {request.question[:100]}... (stream={request.stream})")

        if request.stream:
            return StreamingResponse(
                generate_streaming_response(request.question, request.history),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
            )

        result = rag_workflow.run(request.question, request.history)

        raw_references = result.get("references", [])
        enriched_references = enrich_references_with_urls(raw_references)

        references = [
            DocumentReference(
                document_id=ref.get("document_id", "unknown"),
                type=ref.get("type", "DOCUMENT"),
                description=ref.get("description", None),
                url=ref.get("url", None),
                filename=ref.get("filename", None),
                file_type=ref.get("file_type", None),
                section_path=ref.get("section_path", None),
                page_num=ref.get("page_num", None),
            )
            for ref in enriched_references
        ]

        return ChatResponse(
            answer=result.get("answer", "Lỗi xử lý câu hỏi"),
            references=references,
            status=result.get("status", "ERROR")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(500, f"Internal server error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        db_connected = False
        try:
            db_connected = milvus_client.check_connection()
        except Exception as db_error:
            logger.warning(f"Database check failed: {db_error}")

        workflow_ready = rag_workflow is not None
        url_service_ready = document_url_service.collection is not None

        if db_connected and workflow_ready:
            message = "Hệ thống hoạt động bình thường"
            if url_service_ready:
                message += " (với document URLs)"
            return HealthResponse(status="healthy", message=message, database_connected=True)
        elif workflow_ready and not db_connected:
            return HealthResponse(status="degraded", message="Mất kết nối cơ sở dữ liệu", database_connected=False)
        else:
            return HealthResponse(status="unhealthy", message="Hệ thống gặp sự cố", database_connected=False)

    except Exception as e:
        return HealthResponse(status="unhealthy", message=f"Lỗi: {str(e)}", database_connected=False)


@app.get("/agents")
async def list_agents():
    from config.settings import settings
    return {
        "agents": {
            "SUPERVISOR": "Điều phối chính",
            "FAQ": "Câu hỏi thường gặp / nghiệp vụ nội bộ",
            "RETRIEVER": "Tìm kiếm tài liệu",
            "GRADER": "Đánh giá chất lượng",
            "GENERATOR": "Tạo câu trả lời kèm trích dẫn nguồn (streaming)",
            "NOT_ENOUGH_INFO": "Xử lý thiếu thông tin",
            "CHATTER": "Xử lý cảm xúc",
            "REPORTER": "Báo cáo hệ thống",
            "OTHER": "Yêu cầu ngoài phạm vi"
        },
        "features": {
            "streaming": "enabled",
            "context_aware": "enabled",
            "document_urls": "enabled",
            "inline_citations": "enabled",
            "ngrok_integration": settings.NGROK_PUBLIC_URL is not None
        },
        "status": "ready" if rag_workflow else "not_initialized"
    }


if __name__ == "__main__":
    import uvicorn
    from config.settings import settings
    uvicorn.run(app, host="0.0.0.0", port=settings.RAG_API_PORT, log_level="info")