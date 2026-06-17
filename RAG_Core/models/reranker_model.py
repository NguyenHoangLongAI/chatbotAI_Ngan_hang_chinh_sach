# RAG_Core/models/reranker_model.py
"""
Local reranker dùng BAAI/bge-reranker-v2-m3 qua transformers thuần.
- KHÔNG dùng FlagEmbedding → không có multi-process pool, không spawn subprocess
- Model load 1 lần duy nhất khi khởi động, giữ trên GPU suốt vòng đời app
- Inference ~50ms/batch thay vì ~50s
"""

import os
os.environ.setdefault("HF_HOME", "/mnt/data/nhlong22/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/data/nhlong22/.cache/huggingface")
# Chỉ cho phép thấy cuda:0 — đảm bảo không thư viện nào tự spawn multi-GPU
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class RerankerModel:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"⏳ Loading reranker: {settings.RERANKER_MODEL} ...")

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            self.tokenizer = AutoTokenizer.from_pretrained(settings.RERANKER_MODEL)

            self.model = AutoModelForSequenceClassification.from_pretrained(
                settings.RERANKER_MODEL,
                torch_dtype=torch.float16 if settings.RERANKER_USE_FP16 and self.device.type == "cuda" else torch.float32,
            )
            self.model.to(self.device)
            self.model.eval()

            logger.info(f"✅ Reranker ready on {self.device} (fp16={settings.RERANKER_USE_FP16})")

        except Exception as e:
            logger.error(f"❌ Failed to load reranker model: {e}")
            self.tokenizer = None
            self.model = None

    @property
    def available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @torch.no_grad()
    def rerank(self, query: str, documents: List[str], max_length: int = None) -> List[float]:
        if not self.available:
            raise RuntimeError("Reranker model is not loaded")
        if not documents:
            return []

        max_length = max_length or settings.RERANKER_MAX_LENGTH
        pairs = [[query, doc] for doc in documents]

        scores_all: List[float] = []

        # Xử lý theo batch để tránh OOM với TOP_K lớn
        for i in range(0, len(pairs), settings.RERANKER_BATCH_SIZE):
            batch = pairs[i : i + settings.RERANKER_BATCH_SIZE]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)

            logits = self.model(**inputs).logits.squeeze(-1)
            scores_all.extend(logits.cpu().float().tolist())

        return scores_all

    def rerank_with_index(
        self, query: str, documents: List[str], max_length: int = None
    ) -> List[tuple[int, float]]:
        scores = self.rerank(query, documents, max_length)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return indexed


# Load 1 lần duy nhất khi import
reranker_model = RerankerModel()