# RAG_Core/api/schemas.py - COMPLETE VERSION WITH URL FIELDS

from pydantic import BaseModel
from typing import List, Optional, Union, Literal


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[Union[List[str], List[ChatMessage]]] = []
    stream: Optional[bool] = False


class StreamChunk(BaseModel):
    """Single chunk in streaming response"""
    type: Literal["start", "chunk", "references", "end", "error"]
    content: Optional[str] = None
    references: Optional[List['DocumentReference']] = None
    status: Optional[str] = None


class DocumentReference(BaseModel):
    """
    Document reference with optional URL information

    Fields:
    - document_id: Unique document identifier
    - type: Reference type (FAQ, DOCUMENT, SUPPORT, SYSTEM)
    - description: Document description/content preview
    - url: Public URL to document (NEW)
    - filename: Original filename (NEW)
    - file_type: File extension like .pdf, .docx (NEW)
    """
    document_id: str
    type: str  # FAQ, DOCUMENT, SUPPORT, SYSTEM
    description: Optional[str] = None

    # ===== NEW FIELDS =====
    url: Optional[str] = None  # https://ngrok.../file.pdf
    filename: Optional[str] = None  # file.pdf
    file_type: Optional[str] = None  # .pdf


class ChatResponse(BaseModel):
    answer: str
    references: List[DocumentReference]
    status: str = "SUCCESS"


class HealthResponse(BaseModel):
    status: str
    message: str
    database_connected: bool