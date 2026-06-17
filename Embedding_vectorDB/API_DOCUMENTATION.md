# API Summary - Tóm tắt APIs

## Base URL
```
https://f7a9e2724f33.ngrok-free.app
```

---

## 🔍 **API 0: Health Check**
```
GET /api/v1/health
```

**Đầu vào:** Không  
**Đầu ra:** Trạng thái các service
```json
{
  "status": "healthy",
  "services": {
    "milvus": true,
    "embedding_model": true,
    "chatbot": "ready"
  }
}
```

---

## 📄 **API 1: Process Document** 
```
POST /api/v1/process-document
```

**Đầu vào:** File upload (PDF, Word, Excel, TXT)
```bash
curl -X POST "https://f7a9e2724f33.ngrok-free.app/api/v1/process-document" \
  -F "file=@document.pdf"
```

**Đầu ra:** Markdown content
```json
{
  "status": "success",
  "filename": "document.pdf",
  "markdown_content": "# Title\n## Section\nContent...",
  "processing_info": {
    "file_type": ".pdf",
    "content_length": 1524
  }
}
```

---

## 🔗 **API 2: Embed Markdown**
```
POST /api/v1/embed-markdown
```

**Đầu vào:** Markdown content + document ID
```json
{
  "markdown_content": "# Title\nContent...",
  "document_id": "doc_001"
}
```

**Đầu ra:** Embedding results
```json
{
  "status": "success",
  "document_id": "doc_001",
  "embeddings_count": 5,
  "stored_count": 5,
  "chunks": [...]
}
```

---

## ❓ **API 3: Add FAQ**
```
POST /api/v1/faq/add
```

**Đầu vào:** Câu hỏi + câu trả lời FAQ
```json
{
  "question": "Làm sao để reset mật khẩu?",
  "answer": "Bạn click vào 'Quên mật khẩu' và làm theo hướng dẫn",
  "faq_id": "faq_001"
}
```

**Đầu ra:** Kết quả thêm FAQs
```json
{
  "status": "success",
  "faq_id": "faq_001",
  "question": "Làm sao để reset mật khẩu?",
  "answer": "Bạn click vào 'Quên mật khẩu' và làm theo hướng dẫn",
  "message": "FAQ added successfully"
}
```

---

## 🗑️ **API 4: Delete FAQ**
```
DELETE /api/v1/faq/delete/{faq_id}
```

**Đầu vào:** FAQ ID trong URL
```bash
curl -X DELETE "https://f7a9e2724f33.ngrok-free.app/api/v1/faq/delete/faq_001"
```

**Đầu ra:** Kết quả xóa
```json
{
  "status": "success",
  "faq_id": "faq_001",
  "message": "FAQ deleted successfully"
}
```

---

## 📋 **API 5: Delete Document**
```
DELETE /api/v1/document/delete/{document_id}
```

**Đầu vào:** Document ID trong URL
```bash
curl -X DELETE "https://f7a9e2724f33.ngrok-free.app/api/v1/document/delete/doc_001"
```

**Đầu ra:** Kết quả xóa document
```json
{
  "status": "success",
  "document_id": "doc_001",
  "message": "Document and all its embeddings deleted successfully"
}
```

---

## 🔄 **Workflow cơ bản:**

1. **Upload file** → API 1 → Nhận markdown
2. **Tạo embeddings** → API 2 → Lưu vào vector DB  
3. **Hỏi đáp** → API 3 → Nhận câu trả lời
4. **Quản lý FAQ** → API 4,5 → Thêm/xóa FAQ
5. **Quản lý tài liệu** → API 6 → Xóa document