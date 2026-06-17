# RAG_Core/api/schemas.py

from pydantic import BaseModel
from typing import List, Optional, Union, Literal


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[Union[List[str], List[ChatMessage]]] = []
    stream: Optional[bool] = True


class StreamChunk(BaseModel):
    type: Literal["start", "chunk", "references", "end", "error"]
    content: Optional[str] = None
    references: Optional[List['DocumentReference']] = None
    status: Optional[str] = None


class DocumentReference(BaseModel):
    """
    Document reference — bổ sung section_path / page_num để
    người dùng biết chính xác trích dẫn nằm ở đâu trong tài liệu.
    """
    document_id: str
    type: str  # FAQ, DOCUMENT, SUPPORT, SYSTEM
    description: Optional[str] = None

    url: Optional[str] = None
    filename: Optional[str] = None
    file_type: Optional[str] = None

    # NEW
    section_path: Optional[str] = None
    page_num: Optional[int] = None


class ChatResponse(BaseModel):
    answer: str
    references: List[DocumentReference]
    status: str = "SUCCESS"


class HealthResponse(BaseModel):
    status: str
    message: str
    database_connected: bool