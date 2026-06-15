# RAG Multi-Agent Chatbot System

Hệ thống chatbot RAG (Retrieval-Augmented Generation) sử dụng kiến trúc multi-agent với LangGraph, chạy hoàn toàn local trên GPU với model GPT-OSS-20B và Vietnamese-SBERT embeddings.

## Kiến trúc hệ thống

### Multi-Agent Architecture
- **SUPERVISOR**: Điều phối chính, phân loại yêu cầu và xử lý context
- **FAQ**: Tìm kiếm câu hỏi thường gặp với cross-encoder reranking
- **RETRIEVER**: Tìm kiếm tài liệu từ vector database
- **GRADER**: Đánh giá chất lượng thông tin với BGE Reranker v2-m3
- **GENERATOR**: Tạo câu trả lời từ thông tin đã lọc (hỗ trợ streaming)
- **NOT_ENOUGH_INFO**: Xử lý thiếu thông tin
- **CHATTER**: An ủi cảm xúc tiêu cực khách hàng
- **REPORTER**: Thông báo lỗi/bảo trì hệ thống
- **OTHER**: Xử lý yêu cầu ngoài phạm vi

### Tech Stack

#### Backend Services
- **Vector Database**: Milvus v2.4.4
- **Object Storage**: MinIO
- **Coordination**: etcd v3.5.23
- **Vector DB UI**: Attu

#### AI/ML Models
- **Embedding Model**: keepitreal/vietnamese-sbert (768D)
- **LLM**: GPT-OSS-20B via Ollama 0.12.0
- **Reranker**: BAAI/bge-reranker-v2-m3 (Vietnamese-optimized)
- **OCR**: EasyOCR + Tesseract

#### Document Processing
- **Docling**: v2.17.0 - Advanced document conversion
- **OCR Engines**: EasyOCR v1.7.0, Tesseract OCR
- **Supported Formats**: PDF, DOCX, XLSX, TXT, Images

#### Frameworks & Libraries
- **API Framework**: FastAPI 0.110.0
- **LLM Framework**: LangChain + LangGraph
- **Deep Learning**: PyTorch 2.2.2+cu121
- **Transformers**: Hugging Face Transformers 4.44.2
- **Language**: Python 3.11+

#### Infrastructure
- **Containerization**: Docker & Docker Compose
- **GPU Support**: CUDA 12.1.1 + cuDNN 8

## Cấu trúc dự án

```
.
├── Embedding_vectorDB/          # Document Processing Service
│   ├── config.py               # Cấu hình xử lý tài liệu
│   ├── document_processor.py   # Smart chunking + legacy processing
│   ├── docling_processor.py    # Docling integration
│   ├── embedding_service.py    # Vietnamese SBERT embeddings
│   ├── main.py                 # FastAPI endpoints
│   └── milvus_client.py        # Vector database client
│
├── RAG_Core/                    # RAG Multi-Agent System
│   ├── agents/                 # Agent implementations
│   │   ├── supervisor.py      # Context-aware routing
│   │   ├── faq_agent.py       # FAQ with multi-variant reranking
│   │   ├── retriever_agent.py # Document retrieval
│   │   ├── grader_agent.py    # BGE reranker integration
│   │   ├── generator_agent.py # Streaming response generation
│   │   ├── base_agent.py      # Streaming-enabled base classes
│   │   └── ...
│   ├── models/                # Model wrappers
│   │   ├── embedding_model.py # Vietnamese SBERT
│   │   └── llm_model.py       # Ollama LLM with streaming
│   ├── database/              # Database clients
│   │   └── milvus_client.py   # Milvus operations
│   ├── tools/                 # LangChain tools
│   │   └── vector_search.py   # Search + Vietnamese reranking
│   ├── workflow/              # LangGraph workflow
│   │   └── rag_workflow.py    # Parallel execution + streaming
│   ├── api/                   # FastAPI endpoints
│   │   ├── main.py            # Streaming SSE endpoint
│   │   └── schemas.py         # Request/response models
│   ├── config/                # Configuration
│   │   └── settings.py        # Environment settings
│   ├── utils/                 # Utilities
│   │   ├── context_processor.py # Context extraction
│   │   └── helpers.py
│   └── main.py                # Entry point
│
├── docker-compose.yml          # Multi-service orchestration
├── Dockerfile                  # CUDA + Python environment
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Hướng dẫn cài đặt và chạy

### Yêu cầu hệ thống
- **OS**: Linux (Ubuntu 22.04 recommended)
- **Python**: 3.11+
- **Docker**: 24.0+ & Docker Compose v2
- **GPU**: NVIDIA GPU với ít nhất 16GB VRAM (để chạy GPT-OSS-20B)
- **CUDA**: 12.1+ with cuDNN 8
- **RAM**: Tối thiểu 32GB
- **Storage**: Tối thiểu 100GB free space (cho models và volumes)

### Bước 1: Clone repository

```bash
git clone <repository-url>
cd <project-directory>
```

### Bước 2: Cấu hình môi trường (Optional)

Tạo file `.env` để override default settings:

```bash
# Milvus Configuration
MILVUS_HOST=milvus
MILVUS_PORT=19530

# Ollama Configuration
OLLAMA_URL=http://ollama:11434
LLM_MODEL=gpt-oss:20b

# Embedding Configuration
EMBEDDING_MODEL=keepitreal/vietnamese-sbert
EMBEDDING_DIM=768

# Search Configuration
SIMILARITY_THRESHOLD=0.2
TOP_K=20

# FAQ Optimization
FAQ_VECTOR_THRESHOLD=0.5
FAQ_RERANK_THRESHOLD=0.6
FAQ_TOP_K=10

# Document Grader
DOCUMENT_RERANK_THRESHOLD=0.7

# Support Contact
SUPPORT_PHONE="Phòng vận hành 0904540490 - Phòng kinh doanh:0914616081"
```

### Bước 3: Khởi động hệ thống với Docker Compose

```bash
# Build và khởi động tất cả services
docker-compose up -d

# Xem logs để theo dõi quá trình khởi động
docker-compose logs -f

# Kiểm tra trạng thái containers
docker-compose ps
```

**Services sẽ khởi động theo thứ tự:**
1. etcd (coordination)
2. MinIO (object storage)
3. Milvus (vector database)
4. Attu (Milvus UI)
5. Ollama (LLM server)
6. ollama-init (auto-pull gpt-oss:20b model)
7. document-api (port 8000)
8. rag-api (port 8501)

**Thời gian khởi động dự kiến**: 5-10 phút (phụ thuộc vào network speed khi pull model GPT-OSS-20B ~12GB)

### Bước 4: Xác minh các services

```bash
# Check Milvus
curl http://localhost:9091/healthz

# Check Ollama
curl http://localhost:11434/api/tags

# Check Document API
curl http://localhost:8000/api/v1/health

# Check RAG API
curl http://localhost:8501/health
```

### Bước 5: Truy cập Web UIs

- **Attu (Milvus UI)**: http://localhost:3000
- **MinIO Console**: http://localhost:9001 (minioadmin/minioadmin)
- **Document API Docs**: http://localhost:8000/docs
- **RAG API Docs**: http://localhost:8501/docs

## 📝 Sử dụng hệ thống

### 1. Process Documents (Document API)

#### Process PDF/DOCX/XLSX/TXT
```bash
curl -X POST "http://localhost:8000/api/v1/process-document" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "status": "success",
  "filename": "document.pdf",
  "markdown_content": "# Title\n## Section\n...",
  "processing_info": {
    "file_type": ".pdf",
    "content_length": 1524,
    "processor": "docling+smart_chunker"
  }
}
```

#### Embed Markdown với Smart Chunking
```bash
curl -X POST "http://localhost:8000/api/v1/embed-markdown" \
  -H "Content-Type: application/json" \
  -d '{
    "markdown_content": "# Title\nContent...",
    "document_id": "doc_001",
    "chunk_mode": "smart"
  }'
```

**Chunk Modes:**
- `smart` (recommended): Semantic chunking với context preservation
- `sentence`: Legacy sentence-based chunking
- `legacy`: Backward compatible mode

### 2. Add FAQs
```bash
curl -X POST "http://localhost:8000/api/v1/faq/add" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Làm sao để reset mật khẩu?",
    "answer": "Bạn click vào Quên mật khẩu và làm theo hướng dẫn",
    "faq_id": "faq_001"
  }'
```

### 3. Chat với RAG System

#### Non-Streaming Mode
```bash
curl -X POST "http://localhost:8501/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Làm thế nào để đăng ký dịch vụ?",
    "history": [],
    "stream": false
  }'
```

#### Streaming Mode (SSE)
```bash
curl -X POST "http://localhost:8501/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Làm thế nào để đăng ký dịch vụ?",
    "history": [],
    "stream": true
  }'
```

**Streaming Response Format:**
```
data: {"type":"start","content":null,"references":null,"status":"processing"}

data: {"type":"chunk","content":"Để đăng ký","references":null,"status":null}

data: {"type":"chunk","content":" dịch vụ","references":null,"status":null}

data: {"type":"references","content":null,"references":[...],"status":null}

data: {"type":"end","content":null,"references":null,"status":"SUCCESS"}
```

### 4. Python Client Example

```python
from chat_client import StreamingChatClient

# Initialize client
client = StreamingChatClient("http://localhost:8501/")

# Check health
client.check_health()

# Send streaming message
client.send_message_streaming("Trợ lý ảo là gì?")

# Send non-streaming message
client.send_message_non_streaming("Làm sao sử dụng AI?")

# Interactive mode
client.interactive_mode()
```

## Configuration

### Agent Behavior Tuning

#### FAQ Agent (`config/settings.py`)
```python
# Vector search phase
FAQ_VECTOR_THRESHOLD: float = 0.5  # Lower = more candidates
FAQ_TOP_K: int = 10                # Number of candidates for reranking

# Reranking phase
FAQ_RERANK_THRESHOLD: float = 0.6  # Higher = more confident

# Reranking weights (must sum to 1.0)
FAQ_QUESTION_WEIGHT: float = 0.5   # Question-only matching
FAQ_QA_WEIGHT: float = 0.3         # Question+Answer matching
FAQ_ANSWER_WEIGHT: float = 0.2     # Answer-only matching

# Bonus for consistent scores
FAQ_CONSISTENCY_BONUS: float = 1.1
FAQ_CONSISTENCY_THRESHOLD: float = 0.75
```

#### Document Grader
```python
DOCUMENT_RERANK_THRESHOLD: float = 0.7
```

#### General Search
```python
SIMILARITY_THRESHOLD: float = 0.2
TOP_K: int = 20
```

### Custom Prompts

Chỉnh sửa prompts trong agent files:

- **Supervisor**: `agents/supervisor.py` - Classification prompt
- **FAQ**: `agents/faq_agent.py` - FAQ answering prompt
- **Generator**: `agents/generator_agent.py` - Answer generation prompts
- **Chatter**: `agents/chatter_agent.py` - Emotional support prompt
- **Not Enough Info**: `agents/not_enough_info_agent.py` - Fallback response prompt

### Docling Configuration

```python
# config.py in Embedding_vectorDB
USE_DOCLING: bool = True          # Enable/disable Docling
USE_OCR: bool = True              # Enable OCR for scanned docs
OCR_LANGUAGES: str = "vi,en"      # OCR languages
OCR_USE_GPU: bool = True          # Use GPU for OCR
```

### Smart Chunking Configuration

```python
# document_processor.py
SmartChunker(
    target_chunk_size=500,        # Target tokens per chunk
    min_chunk_size=200,           # Minimum chunk size
    max_chunk_size=800,           # Maximum chunk size
    overlap_size=100              # Overlap between chunks
)
```

## Development & Testing

### Run Document API Tests
```bash
cd Embedding_vectorDB
python test_API.py
```

### Run RAG Chat Client
```bash
cd RAG_Core
python chat_client.py

# Or test specific question
python chat_client.py "Trợ lý ảo là gì?"
```

### Monitor Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f rag-api
docker-compose logs -f document-api
docker-compose logs -f ollama
```

### Debug Mode

Enable debug logging in `api/main.py`:
```python
logging.basicConfig(
    level=logging.DEBUG,  # Change from INFO to DEBUG
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Troubleshooting

### Ollama Model Not Loading
```bash
# Check if model exists
docker exec ollama ollama list

# Manually pull model
docker exec ollama ollama pull gpt-oss:20b
```

### Milvus Connection Issues
```bash
# Check Milvus health
curl http://localhost:9091/healthz

# Restart Milvus
docker-compose restart milvus
```

### GPU Not Detected
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Check Docker GPU support
docker info | grep -i gpu
```

### Document Processing Fails
```bash
# Check document-api logs
docker-compose logs document-api

# Verify Tesseract installation
docker exec document-api tesseract --version
```

### Slow Response Times
- Check GPU utilization: `nvidia-smi`
- Reduce `TOP_K` in settings
- Increase `SIMILARITY_THRESHOLD` to filter more aggressively
- Use FAQ mode instead of full RAG when possible

## Performance Optimization

### Parallel Execution
The system uses parallel execution for:
- Supervisor classification
- FAQ search
- Document retrieval

Adjust worker threads in `rag_workflow.py`:
```python
self.executor = ThreadPoolExecutor(
    max_workers=3,  # Increase for more parallelism
    thread_name_prefix="RAG-Worker"
)
```

### Context Caching
Context processor uses LRU cache:
```python
ContextProcessor(
    max_context_length=500,
    cache_size=10  # Increase for more caching
)
```

### Streaming Latency
Adjust streaming chunk delays in `api/main.py`:
```python
await asyncio.sleep(0.001)  # Reduce for faster streaming
```

## Security Considerations

- Change default MinIO credentials in production
- Use environment variables for sensitive configs
- Enable authentication for Milvus in production
- Implement rate limiting on API endpoints
- Use HTTPS in production deployment

## Production Deployment

### Resource Requirements
- **CPU**: 8+ cores
- **RAM**: 64GB+
- **GPU**: NVIDIA A100 or similar (24GB+ VRAM)
- **Storage**: 500GB+ SSD

### Docker Compose Production Override
Create `docker-compose.prod.yml`:
```yaml
version: '3.8'

services:
  rag-api:
    restart: always
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Deploy:
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## API Documentation

Full API documentation available at:
- Document API: http://localhost:8000/docs
- RAG API: http://localhost:8501/docs

## Key Features Summary

**Smart Document Processing**
- Docling integration for high-quality conversion
- Semantic chunking with context preservation
- Multi-format support (PDF, DOCX, XLSX, TXT)

**Vietnamese-Optimized RAG**
- Vietnamese SBERT embeddings (768D)
- BGE Reranker v2-m3 for Vietnamese
- Multi-variant FAQ matching

**Streaming Support**
- Server-Sent Events (SSE) streaming
- True async streaming from LLM
- Progressive answer generation

**Multi-Agent Architecture**
- Parallel execution (Supervisor + FAQ + Retriever)
- Context-aware routing
- Specialized agents for different scenarios

**Production Ready**
- Docker Compose orchestration
- Health checks and auto-restart
- GPU acceleration
- Comprehensive logging

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Your License Here]

---

** Chúc bạn thành công với RAG Multi-Agent Chatbot!**

For issues and questions, please create an issue on GitHub or contact the maintainers.
