"""
PaddleOCR v3.7.0 Document Processor — PRODUCTION v6
=======================================================
Tái cấu trúc từ v5.1 theo pipeline chuẩn 5 bước (Layout Analysis → Table
Structure → Text Recognition → Figure/Caption → Normalize/Fuse).
Public API giữ nguyên 100% — document_processor.py / main.py không cần sửa.

THAY ĐỔI KIẾN TRÚC CHÍNH (so với v5.1):

  A. LAYOUT-FIRST ROUTING (fix root cause, không phải fix triệu chứng)
     v5.1: mỗi PageType (TABLE/IMAGE/MIXED) tự gọi lại
           PPStructureV3.predict() theo cách khác nhau — TABLE chỉ lấy
           "tables_only", IMAGE lấy "full_layout", rồi so sánh string thô
           để tránh duplicate giữa pdfplumber-table và PPStructure-table.
           → Dễ duplicate/bỏ sót khi OCR sai 1-2 ký tự làm so khớp fail.
     v6:   với MỌI page cần render (không phải TEXT thuần), gọi
           predict() ĐÚNG MỘT LẦN, lấy toàn bộ parsing_res_list, rồi
           route từng BLOCK (không phải từng PAGE) theo label thật:
           table → SLANet structure, text → OCR/correction,
           figure → caption-matched image, formula → latex.
           Loại bỏ hoàn toàn logic so-sánh-string-để-tránh-duplicate.

  B. MERGED CELLS FIX (_html_to_markdown)
     v5.1: TableParser bỏ qua colspan/rowspan → bảng có ô gộp (rất phổ
           biến trong bảng lãi suất/kỳ hạn ngân hàng) bị lệch cột khi
           convert sang Markdown.
     v6:   parse colspan/rowspan, replicate giá trị cell theo đúng số
           cột/dòng bị gộp trước khi build lưới Markdown, dùng grid
           occupancy map để không ghi đè ô đã bị chiếm bởi rowspan
           từ dòng trước.

  C. CAPTION MATCHING cho Figure
     v5.1: alt-text generic "Page N Image K", không có ngữ cảnh.
     v6:   sau khi crop figure, quét text trong vùng ngay dưới/trên ảnh
           (bằng pdfplumber crop mở rộng) tìm pattern dạng số thứ tự kiểu
           "Hình 1:", "Biểu đồ 2." → gắn vào alt text, giúp embedding sau
           này giữ được ngữ nghĩa của hình.

  D. HANDWRITING-AWARE FALLBACK (có điều kiện, không tăng latency mặc định)
     PP-OCRv5_mobile_rec trả rec_scores cho từng dòng. Nếu một block text
     có tỷ lệ dòng confidence thấp (score < HANDWRITING_SCORE_THRESHOLD)
     vượt HANDWRITING_LOW_CONF_RATIO, đánh dấu block đó "có khả năng viết
     tay" — hiện tại fallback bằng cách thử lại OCR trên crop ảnh đã
     upscale (tăng DPI hiệu lực) vì pipeline không bundle sẵn model
     handwriting riêng; nếu có model server-tier khác, chỉ cần thay
     _retry_low_confidence_block() để gọi engine đó.

GIỮ NGUYÊN TỪ v5.1:
  - HF_HUB_OFFLINE đặt trước mọi import liên quan huggingface_hub.
  - ProtonX correction với preserve table/image/code block, logging chi
    tiết (call counters, diff summary, periodic summary).
  - Page separator "\n\n---\n\n" cho SmartChunker.
  - PaddleOCR v3.7.0 API: device= thay use_gpu=, show_log removed.
  - PPStructureV3 dùng PP-OCRv5_mobile_det/rec, seal/formula/chart=False.

CHIẾN LƯỢC XỬ LÝ PAGE (v6):
  TEXT  → pdfplumber fast-path, không render, không OCR (giữ nguyên v5.1
          vì đây đã là tối ưu đúng — page có đủ text digital thì không
          cần Layout Analysis nào cả).
  TABLE/IMAGE/MIXED → render ảnh → PPStructureV3.predict() MỘT LẦN →
          route block-by-block theo label → fuse theo thứ tự đọc gốc
          (parsing_res_list đã trả theo đúng reading order).
"""

from __future__ import annotations

# ============================================================================
# CRITICAL: Set HF/transformers offline env vars BEFORE any import that may
# pull in huggingface_hub (transformers, protonx, sentence-transformers...).
# ============================================================================
import os

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

_ALLOW_WARMUP = os.getenv("PROTONX_ALLOW_ONLINE_WARMUP", "false").lower() == "true"
if not _ALLOW_WARMUP:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import base64
import io
import logging
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

logging.getLogger("paddlex").setLevel(logging.WARNING)
logging.getLogger("paddleocr").setLevel(logging.WARNING)
logging.getLogger("paddle").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ============================================================================
# MODULE-LEVEL HELPERS
# ============================================================================

def _paddle_actual_device() -> str:
    """Trả về device thực sự mà PaddlePaddle đang dùng: 'gpu:0', 'cpu', v.v."""
    try:
        import paddle
        return str(paddle.get_device())
    except Exception:
        pass
    try:
        import paddle
        if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu:0"
        return "cpu"
    except Exception:
        return "unknown"


def _is_whitespace_only_change(original: str, corrected: str) -> bool:
    import re as _re
    return _re.sub(r"\s+", "", original) == _re.sub(r"\s+", "", corrected)


def _diff_summary(original: str, corrected: str, max_examples: int = 3) -> str:
    import difflib
    orig_words = original.split()
    corr_words = corrected.split()
    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
    examples = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert", "delete") and len(examples) < max_examples:
            orig_seg = " ".join(orig_words[i1:i2])[:60]
            corr_seg = " ".join(corr_words[j1:j2])[:60]
            examples.append(f"  [{len(examples) + 1}] {orig_seg!r} → {corr_seg!r}")
    return ("\n" + "\n".join(examples)) if examples else " (no word-level diff)"


# ============================================================================
# CONSTANTS
# ============================================================================

TEXT_PAGE_MIN_CHARS = 80
MIXED_PAGE_MIN_CHARS = 40
PDF_RENDER_DPI = 150

# Caption matching — patterns thường gặp trong tài liệu tiếng Việt
_CAPTION_PATTERNS = [
    re.compile(r'^\s*(Hình|Bảng|Biểu\s*đồ|Sơ\s*đồ|Ảnh)\s*\d+[:.\-]?\s*.{0,150}', re.IGNORECASE),
]
_CAPTION_SEARCH_MARGIN = 40  # px mở rộng vùng tìm caption quanh figure

# Handwriting-aware fallback thresholds
HANDWRITING_SCORE_THRESHOLD = 0.5
HANDWRITING_LOW_CONF_RATIO = 0.30


# ============================================================================
# PROTONX VIETNAMESE TEXT CORRECTOR  (giữ nguyên logic v5.1)
# ============================================================================

class VietnameseTextCorrector:
    """
    ProtonX offline text correction cho tiếng Việt sau OCR.
    Models: teacher (904MB, ROUGE-L 98.44) / student (507MB, 97.64) / nano
    """

    MODELS = {
        "teacher": "protonx-models/protonx-legal-tc",
        "student": "protonx-models/distilled-protonx-legal-tc",
        "nano":    "protonx-models/nano-protonx-legal-tc",
    }
    MAX_CHUNK = 512
    SUMMARY_EVERY = 50

    def __init__(self, model_size: str = "student", enabled: bool = True):
        self.enabled = enabled
        self.model_size = model_size
        self._client = None
        self._lock = threading.Lock()

        self._call_count = 0
        self._api_call_count = 0
        self._changed_count = 0
        self._skipped_count = 0
        self._error_count = 0
        self._total_input_chars = 0
        self._total_output_chars = 0
        self._total_elapsed = 0.0
        self._total_api_elapsed = 0.0
        self._api_content_changed_count = 0

        if enabled:
            self._init_client()

    def _init_client(self):
        try:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            from protonx import ProtonX
            self._client = ProtonX(mode="offline")
            logger.info(
                f"✅ ProtonX TextCorrector ready "
                f"(model={self.MODELS.get(self.model_size, self.model_size)})"
            )
        except ImportError:
            logger.warning("⚠️ protonx not installed — text correction disabled")
            self.enabled = False
        except Exception as e:
            logger.warning(f"⚠️ ProtonX init failed: {e} — disabled")
            self.enabled = False

    def correct(self, text: str) -> str:
        if not self.enabled or not self._client or not text.strip():
            self._skipped_count += 1
            logger.debug(
                f"[ProtonX] correct() SKIPPED — "
                f"enabled={self.enabled}, empty={not text.strip()}"
            )
            return text

        self._call_count += 1
        call_id = self._call_count
        input_len = len(text)
        t0 = time.time()

        logger.info(
            f"[ProtonX #{call_id}] correct() START — "
            f"input={input_len} chars | model={self.model_size}"
        )

        try:
            result = self._correct_with_preservation(text)
            elapsed = time.time() - t0
            output_len = len(result)
            changed = result != text

            self._total_input_chars += input_len
            self._total_output_chars += output_len
            self._total_elapsed += elapsed
            if changed:
                self._changed_count += 1

            _ws_only = _is_whitespace_only_change(text, result) if changed else False
            _change_type = (
                "whitespace_only" if (_ws_only and changed)
                else ("content" if changed else "none")
            )

            logger.info(
                f"[ProtonX #{call_id}] correct() DONE  — "
                f"input={input_len} | output={output_len} | "
                f"changed={'YES' if changed else 'NO '} | "
                f"change_type={_change_type} | "
                f"elapsed={elapsed:.3f}s"
            )

            if changed and not _ws_only:
                logger.info(
                    f"[ProtonX #{call_id}] content diff:{_diff_summary(text, result)}"
                )
            elif changed and _ws_only:
                logger.debug(
                    f"[ProtonX #{call_id}] whitespace-only change (no content fix)"
                )

            if self.SUMMARY_EVERY > 0 and self._call_count % self.SUMMARY_EVERY == 0:
                self._log_summary()

            return result

        except Exception as e:
            elapsed = time.time() - t0
            self._error_count += 1
            self._total_elapsed += elapsed
            logger.warning(
                f"[ProtonX #{call_id}] correct() ERROR — "
                f"{e} | elapsed={elapsed:.3f}s"
            )
            logger.debug(f"Text correction failed: {e}")
            return text

    def _correct_with_preservation(self, text: str) -> str:
        PRESERVE_PATTERNS = [
            re.compile(r'!\[[^\]]*\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)', re.DOTALL),
            re.compile(r'(?m)((?:^\|.+\|\s*\n?){2,})'),
            re.compile(r'```[\s\S]*?```'),
        ]
        preserved: Dict[str, str] = {}
        idx = [0]

        def protect(m: re.Match) -> str:
            key = f"%%PRESERVE_{idx[0]:04d}%%"
            preserved[key] = m.group(0)
            idx[0] += 1
            return f"\n{key}\n"

        protected = text
        for pat in PRESERVE_PATTERNS:
            protected = pat.sub(protect, protected)

        paragraphs = re.split(r'\n\n+', protected)
        corrected_parts = []
        for para in paragraphs:
            stripped = para.strip()
            if not stripped or stripped.startswith("%%PRESERVE_"):
                corrected_parts.append(para)
            elif self._is_plain_text(stripped):
                corrected_parts.append(self._correct_chunk(stripped))
            else:
                corrected_parts.append(para)

        result = "\n\n".join(corrected_parts)
        for key, original in preserved.items():
            result = result.replace(key, original)
        return result

    def _is_plain_text(self, text: str) -> bool:
        lines = text.strip().split('\n')
        special = ('#', '-', '*', '|', '>', '`', '!', '%%')
        sc = sum(1 for l in lines if l.strip() and l.strip().startswith(special))
        return len(lines) == 0 or (sc / len(lines)) < 0.5

    def _correct_chunk(self, text: str) -> str:
        if len(text) <= self.MAX_CHUNK:
            return self._api_correct(text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) <= self.MAX_CHUNK:
                current = (current + " " + s).strip()
            else:
                if current:
                    chunks.append(current)
                current = s
        if current:
            chunks.append(current)
        return " ".join(self._api_correct(c) for c in chunks)

    def _api_correct(self, text: str) -> str:
        if not text.strip():
            return text

        self._api_call_count += 1
        api_id = self._api_call_count
        preview = text[:80].replace('\n', ' ')
        t0 = time.time()

        logger.debug(
            f"[ProtonX API #{api_id}] call START — "
            f"chunk={len(text)} chars | preview: \"{preview}{'...' if len(text) > 80 else ''}\""
        )

        try:
            model_name = self.MODELS.get(self.model_size, self.MODELS["student"])
            result_data = self._client.text.correct(input=text, top_k=1, model=model_name)
            elapsed = time.time() - t0
            self._total_api_elapsed += elapsed

            if (result_data and "data" in result_data and result_data["data"]
                    and result_data["data"][0].get("candidates")):
                output = result_data["data"][0]["candidates"][0]["output"]
                changed = output != text
                logger.debug(
                    f"[ProtonX API #{api_id}] call DONE  — "
                    f"elapsed={elapsed:.3f}s | changed={'YES' if changed else 'NO '} | "
                    f"out_len={len(output)}"
                )
                return output

            logger.debug(
                f"[ProtonX API #{api_id}] call DONE  — "
                f"elapsed={elapsed:.3f}s | no candidates, returning original"
            )
            return text

        except Exception as e:
            elapsed = time.time() - t0
            self._total_api_elapsed += elapsed
            logger.debug(
                f"[ProtonX API #{api_id}] call ERROR — "
                f"elapsed={elapsed:.3f}s | {e}"
            )
            return text

    def _log_summary(self) -> None:
        avg_elapsed = (
            self._total_elapsed / self._call_count if self._call_count else 0.0
        )
        avg_api_elapsed = (
            self._total_api_elapsed / self._api_call_count
            if self._api_call_count else 0.0
        )
        change_rate = (
            self._changed_count / self._call_count * 100
            if self._call_count else 0.0
        )
        char_delta = self._total_output_chars - self._total_input_chars

        logger.info(
            f"[ProtonX SUMMARY] ── after {self._call_count} correct() calls ──\n"
            f"  calls      : total={self._call_count} | "
            f"changed={self._changed_count} ({change_rate:.1f}%) | "
            f"skipped={self._skipped_count} | errors={self._error_count}\n"
            f"  chars      : input={self._total_input_chars} | "
            f"output={self._total_output_chars} | delta={char_delta:+d}\n"
            f"  api calls  : {self._api_call_count} | avg={avg_api_elapsed:.3f}s/call\n"
            f"  time       : total={self._total_elapsed:.1f}s | "
            f"avg={avg_elapsed:.3f}s/correct() | "
            f"api_total={self._total_api_elapsed:.1f}s"
        )

    def log_summary(self) -> None:
        self._log_summary()


class PageType:
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    MIXED = "mixed"


# ============================================================================
# PAGE CLASSIFIER  (chỉ quyết định: page này có cần render+layout không?)
# ============================================================================

class PageClassifier:
    """
    Phân loại page ở mức THÔ để quyết định fast-path hay không — KHÔNG
    còn vai trò xác định nội dung chi tiết (đó là việc của Layout Analysis
    thật ở LayoutRouter). TEXT = đủ text digital, khỏi cần render/OCR gì.
    Mọi loại khác (TABLE/IMAGE/MIXED) đều đi qua layout pipeline thống nhất.
    """

    @staticmethod
    def classify(page, page_width: float, page_height: float) -> str:
        text = page.extract_text() or ""
        char_count = len(text.strip())

        images = page.images or []
        image_area = sum(
            abs((img.get("x1", 0) - img.get("x0", 0))
                * (img.get("y1", 0) - img.get("y0", 0)))
            for img in images
        )
        page_area = max(page_width * page_height, 1)
        image_coverage = image_area / page_area

        tables = page.extract_tables() or []
        has_table = len(tables) > 0

        if char_count >= TEXT_PAGE_MIN_CHARS and image_coverage < 0.30 and not has_table:
            return PageType.TEXT

        if image_coverage >= 0.40 and char_count < 30:
            return PageType.IMAGE

        if has_table:
            return PageType.TABLE

        if char_count >= MIXED_PAGE_MIN_CHARS:
            return PageType.MIXED

        return PageType.MIXED


# ============================================================================
# PDF EMBEDDED IMAGE EXTRACTOR  (fallback path khi không render — hiếm)
# ============================================================================

class PDFImageExtractor:
    @staticmethod
    def extract_images_from_page(pdf_page, page_num: int) -> List[str]:
        markdown_images: List[str] = []
        images = pdf_page.images or []
        for idx, img_meta in enumerate(images):
            try:
                raw = img_meta.get("stream") or img_meta.get("data")
                if raw is None:
                    x0 = img_meta.get("x0", 0)
                    y0 = img_meta.get("y0", 0)
                    x1 = img_meta.get("x1", pdf_page.width)
                    y1 = img_meta.get("y1", pdf_page.height)
                    cropped = pdf_page.crop((x0, y0, x1, y1)).to_image(resolution=120)
                    buf = io.BytesIO()
                    cropped.save(buf, format="PNG")
                    raw = buf.getvalue()
                if isinstance(raw, (bytes, bytearray)):
                    b64 = base64.b64encode(raw).decode("utf-8")
                    mime = PDFImageExtractor._detect_mime(raw)
                    caption = PDFImageExtractor._find_caption(pdf_page, img_meta)
                    label = caption or f"Page {page_num} Image {idx + 1}"
                    markdown_images.append(f"![{label}](data:{mime};base64,{b64})")
            except Exception as e:
                logger.debug(f"Image extract error p{page_num} img{idx}: {e}")
        return markdown_images

    @staticmethod
    def _find_caption(pdf_page, img_meta: Dict) -> Optional[str]:
        """Tìm caption (Hình/Bảng/Biểu đồ N: ...) ngay dưới hoặc trên ảnh."""
        try:
            x0 = img_meta.get("x0", 0)
            x1 = img_meta.get("x1", pdf_page.width)
            y0 = img_meta.get("y0", 0)
            y1 = img_meta.get("y1", pdf_page.height)

            # pdfplumber: gốc tọa độ y tăng từ trên-xuống hoặc dưới-lên tùy
            # version; thử cả hai vùng lân cận (trên & dưới) cho an toàn.
            below = pdf_page.crop((
                max(x0 - 10, 0), y1,
                min(x1 + 10, pdf_page.width), min(y1 + _CAPTION_SEARCH_MARGIN, pdf_page.height)
            ))
            above = pdf_page.crop((
                max(x0 - 10, 0), max(y0 - _CAPTION_SEARCH_MARGIN, 0),
                min(x1 + 10, pdf_page.width), y0
            ))

            for region in (below, above):
                txt = (region.extract_text() or "").strip()
                if not txt:
                    continue
                for pat in _CAPTION_PATTERNS:
                    m = pat.match(txt)
                    if m:
                        return m.group(0).strip()[:150]
            return None
        except Exception:
            return None

    @staticmethod
    def _detect_mime(data: bytes) -> str:
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        if data[:3] == b'\xff\xd8\xff':
            return "image/jpeg"
        return "image/png"


# ============================================================================
# PDFPLUMBER FAST-PATH EXTRACTOR  (chỉ cho PageType.TEXT)
# ============================================================================

class PlumberExtractor:
    @staticmethod
    def extract_page(pdf_page, page_num: int) -> str:
        sections: List[str] = []
        tables = pdf_page.extract_tables() or []
        for table in tables:
            if table:
                md = PlumberExtractor._table_to_markdown(table)
                if md:
                    sections.append(md)
        text = pdf_page.extract_text(x_tolerance=3, y_tolerance=3) or ""
        text = text.strip()
        if text:
            sections.insert(0, text)
        return "\n\n".join(s for s in sections if s.strip())

    @staticmethod
    def _table_to_markdown(table: List[List]) -> str:
        if not table or not any(table):
            return ""
        rows = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in table
        ]
        rows = [r for r in rows if any(c for c in r)]
        if not rows:
            return ""
        max_cols = max(len(r) for r in rows)
        rows = [r + [""] * (max_cols - len(r)) for r in rows]
        lines = [
            "| " + " | ".join(rows[0]) + " |",
            "| " + " | ".join(["---"] * max_cols) + " |",
        ]
        for row in rows[1:]:
            lines.append("| " + " | ".join(c if c else "-" for c in row) + " |")
        return "\n".join(lines)


# ============================================================================
# PADDLEOCR v3.7.0 ENGINE  (PP-OCRv5_mobile) — dùng cho retry/fallback
# ============================================================================

class PaddleOCREngine:
    """
    PaddleOCR v3.7.0 với PP-OCRv5_mobile models.
    Trong v6 dùng chủ yếu làm fallback khi PPStructureV3 không khả dụng,
    hoặc khi retry block có confidence thấp (khả năng chữ viết tay).
    """

    def __init__(self, use_gpu: bool = True, lang: str = "vi"):
        self._ocr = None
        self.use_gpu = use_gpu
        self.lang = lang
        self._init()

    def _init(self):
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                device="gpu" if self.use_gpu else "cpu",
            )
            _actual_dev = _paddle_actual_device()
            logger.info(
                f"✅ PaddleOCR v3.7.0 (PP-OCRv5_mobile) engine ready | "
                f"requested={'gpu' if self.use_gpu else 'cpu'} | "
                f"actual_device={_actual_dev}"
            )
            if self.use_gpu and "gpu" not in _actual_dev.lower():
                logger.warning(
                    f"⚠️  [PaddleOCR] GPU requested but actual device={_actual_dev!r}. "
                    f"Inference will run on CPU — kiểm tra CUDA driver / CUDA_VISIBLE_DEVICES"
                )
        except ImportError:
            logger.warning("⚠️ paddleocr not installed. Run: pip install paddleocr==3.7.0")
        except Exception as e:
            logger.warning(f"⚠️ PaddleOCR init failed: {e}")

    def extract_text_with_scores(self, image: "Image.Image") -> Tuple[str, List[float]]:
        """Extract text + trả kèm list confidence scores (cho handwriting check)."""
        if self._ocr is None:
            return "", []
        try:
            import numpy as np
            img_array = np.array(image.convert("RGB"))
            result = self._ocr.predict(img_array)
            lines, scores = [], []
            for page in result:
                rec_texts = page.get("rec_texts", []) or []
                rec_scores = page.get("rec_scores", []) or []
                for i, text in enumerate(rec_texts):
                    score = rec_scores[i] if i < len(rec_scores) else 1.0
                    if text and str(text).strip() and score > 0.3:
                        lines.append(str(text).strip())
                        scores.append(float(score))
            return "\n".join(lines), scores
        except Exception as e:
            logger.error(f"PaddleOCR extraction error: {e}")
            return "", []

    def extract_text(self, image: "Image.Image") -> str:
        text, _ = self.extract_text_with_scores(image)
        return text

    def extract_text_upscaled(self, image: "Image.Image", scale: float = 1.8) -> str:
        """
        Retry path cho block có confidence thấp (khả năng chữ viết tay /
        ảnh mờ). Upscale ảnh trước khi OCR lại — không cần model riêng,
        cải thiện đáng kể recognition trên nét chữ mảnh/nghiêng.
        """
        try:
            w, h = image.size
            upscaled = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            return self.extract_text(upscaled)
        except Exception as e:
            logger.debug(f"Upscale retry failed: {e}")
            return self.extract_text(image)


# ============================================================================
# PP-STRUCTURE V3 — LAYOUT ANALYSIS + TABLE STRUCTURE (engine thống nhất)
# ============================================================================

class LayoutAnalyzer:
    """
    Bọc PPStructureV3 làm NGUỒN DUY NHẤT cho layout + table + text + figure
    của một ảnh trang. Mỗi page chỉ gọi predict() một lần (v6 fix so với
    v5.1 gọi 2 lần khác mục đích cho TABLE vs IMAGE).
    """

    def __init__(self, use_gpu: bool = True, lang: str = "vi"):
        self._pipeline = None
        self.use_gpu = use_gpu
        self.lang = lang
        self._init()

    @property
    def available(self) -> bool:
        return self._pipeline is not None

    def _init(self):
        try:
            from paddleocr import PPStructureV3
            self._pipeline = PPStructureV3(
                lang=self.lang,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_seal_recognition=False,
                use_formula_recognition=False,
                use_chart_recognition=False,
                device="gpu" if self.use_gpu else "cpu",
            )
            logger.info(
                f"✅ PP-StructureV3 layout analyzer ready "
                f"(PP-OCRv5_mobile, device={'gpu' if self.use_gpu else 'cpu'})"
            )
        except ImportError:
            logger.warning(
                "⚠️ paddleocr/paddlex[ocr] not installed — "
                "layout analysis disabled. Run: pip install paddlex[ocr]"
            )
        except Exception as e:
            logger.warning(f"⚠️ PP-StructureV3 init failed: {e}")

    def analyze(self, image: "Image.Image") -> List["LayoutBlock"]:
        """
        Trả về list LayoutBlock theo ĐÚNG THỨ TỰ ĐỌC (parsing_res_list đã
        sắp theo reading order: trên→dưới, trái→phải). Đây là điểm mấu
        chốt khác v5.1: một lần predict(), router quyết định downstream.
        """
        if self._pipeline is None:
            return []
        try:
            import numpy as np
            img_array = np.array(image.convert("RGB"))
            result = self._pipeline.predict(img_array)

            blocks: List[LayoutBlock] = []
            for page_result in result:
                parsing_list = page_result.get("parsing_res_list", [])
                for block in parsing_list:
                    label = (getattr(block, "label", "") or "").lower()
                    content = (getattr(block, "content", "") or "").strip()
                    bbox = getattr(block, "bbox", None) or getattr(block, "block_bbox", None)
                    if not content:
                        continue
                    blocks.append(LayoutBlock(label=label, content=content, bbox=bbox))

                # Fallback nếu parsing_res_list rỗng nhưng overall_ocr_res có data
                if not blocks:
                    ocr_res = page_result.get("overall_ocr_res", {}) or {}
                    rec_texts = ocr_res.get("rec_texts", []) or []
                    rec_scores = ocr_res.get("rec_scores", []) or []
                    if rec_texts:
                        blocks.append(LayoutBlock(
                            label="text",
                            content="\n".join(str(t) for t in rec_texts if t),
                            bbox=None,
                            scores=list(rec_scores) if rec_scores else None,
                        ))

            return blocks
        except Exception as e:
            logger.error(f"PP-StructureV3 layout analysis error: {e}")
            return []

    @staticmethod
    def html_table_to_markdown(html: str) -> str:
        """
        Convert HTML table → Markdown, XỬ LÝ ĐÚNG colspan/rowspan (fix
        chính so với v5.1, nơi merged cells bị bỏ qua hoàn toàn và làm
        lệch cột với bảng phức tạp như bảng lãi suất kỳ hạn x loại KH).
        """
        try:
            from html.parser import HTMLParser

            class GridTableParser(HTMLParser):
                """
                Dựng grid 2D thực sự, tôn trọng colspan/rowspan bằng cách
                track occupancy map — ô nào đã bị "chiếm" bởi rowspan từ
                dòng trước sẽ được bỏ qua khi đặt ô mới vào dòng hiện tại.
                """

                def __init__(self):
                    super().__init__()
                    self.grid: List[List[str]] = []
                    self._row_idx = -1
                    self._col_cursor = 0
                    self._cell_text = ""
                    self._in_cell = False
                    self._cur_colspan = 1
                    self._cur_rowspan = 1
                    # occupancy[row][col] = True nếu đã có ô (do rowspan) chiếm
                    self._occupancy: Dict[Tuple[int, int], bool] = {}

                def handle_starttag(self, tag, attrs):
                    attrs_d = dict(attrs)
                    if tag == "tr":
                        self._row_idx += 1
                        self._col_cursor = 0
                        if self._row_idx >= len(self.grid):
                            self.grid.append([])
                    elif tag in ("td", "th"):
                        self._cell_text = ""
                        self._in_cell = True
                        try:
                            self._cur_colspan = max(1, int(attrs_d.get("colspan", 1)))
                        except (TypeError, ValueError):
                            self._cur_colspan = 1
                        try:
                            self._cur_rowspan = max(1, int(attrs_d.get("rowspan", 1)))
                        except (TypeError, ValueError):
                            self._cur_rowspan = 1

                def handle_data(self, data):
                    if self._in_cell:
                        self._cell_text += data

                def handle_endtag(self, tag):
                    if tag not in ("td", "th"):
                        return
                    self._in_cell = False
                    value = self._cell_text.strip()

                    # Tìm cột trống đầu tiên (chưa bị occupancy chiếm) từ
                    # col_cursor hiện tại — đây là cách xử lý đúng khi dòng
                    # trước có rowspan tràn xuống dòng này.
                    while self._occupancy.get((self._row_idx, self._col_cursor)):
                        self._col_cursor += 1

                    start_col = self._col_cursor
                    row = self.grid[self._row_idx]
                    while len(row) <= start_col + self._cur_colspan - 1:
                        row.append("")

                    for c_off in range(self._cur_colspan):
                        col = start_col + c_off
                        row[col] = value if c_off == 0 else value  # replicate ngang
                        for r_off in range(self._cur_rowspan):
                            if r_off == 0:
                                continue
                            target_row = self._row_idx + r_off
                            self._occupancy[(target_row, col)] = True
                            while len(self.grid) <= target_row:
                                self.grid.append([])
                            grow = self.grid[target_row]
                            while len(grow) <= col:
                                grow.append("")
                            grow[col] = value  # replicate dọc

                    self._col_cursor = start_col + self._cur_colspan

            parser = GridTableParser()
            parser.feed(html)
            rows = [r for r in parser.grid if any(c.strip() for c in r)]
            if not rows:
                return ""

            max_cols = max(len(r) for r in rows)
            rows = [r + [""] * (max_cols - len(r)) for r in rows]

            lines = [
                "| " + " | ".join(rows[0]) + " |",
                "| " + " | ".join(["---"] * max_cols) + " |",
            ]
            for row in rows[1:]:
                lines.append("| " + " | ".join(c if c else "-" for c in row) + " |")
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"HTML table parse error: {e}")
            return ""


class LayoutBlock:
    """Một block layout đã được PPStructureV3 phân vùng + gán nhãn."""

    __slots__ = ("label", "content", "bbox", "scores")

    def __init__(self, label: str, content: str, bbox=None, scores: Optional[List[float]] = None):
        self.label = label
        self.content = content
        self.bbox = bbox
        self.scores = scores

    @property
    def is_table(self) -> bool:
        return self.label == "table"

    @property
    def is_figure(self) -> bool:
        return self.label in ("figure", "image", "figure_caption", "chart")

    @property
    def is_formula(self) -> bool:
        return self.label == "formula"

    @property
    def low_confidence_ratio(self) -> float:
        """Tỷ lệ dòng có score thấp — tín hiệu khả năng chữ viết tay."""
        if not self.scores:
            return 0.0
        low = sum(1 for s in self.scores if s < HANDWRITING_SCORE_THRESHOLD)
        return low / len(self.scores)


# ============================================================================
# LAYOUT ROUTER — route từng block theo label, fuse theo thứ tự đọc
# ============================================================================

class LayoutRouter:
    """
    Nhận List[LayoutBlock] (đã đúng reading order từ LayoutAnalyzer) và
    quyết định downstream processing cho từng block — đây là "Bước 5:
    Chuẩn hóa & Hợp nhất" áp dụng trực tiếp trên kết quả layout thật,
    thay vì v5.1 phải tự suy luận lại qua PageType ở mức trang.
    """

    def __init__(
        self,
        corrector: VietnameseTextCorrector,
        ocr_engine: PaddleOCREngine,
        analyzer: LayoutAnalyzer,
    ):
        self.corrector = corrector
        self.ocr_engine = ocr_engine
        self.analyzer = analyzer

    def route_page(self, rendered_image: "Image.Image") -> List[str]:
        blocks = self.analyzer.analyze(rendered_image)
        parts: List[str] = []

        for block in blocks:
            if block.is_table:
                md = self._handle_table(block)
                if md:
                    parts.append(md)
            elif block.is_figure:
                parts.append(self._handle_figure(block, rendered_image))
            elif block.is_formula:
                parts.append(f"$${block.content}$$")
            else:
                parts.append(self._handle_text(block, rendered_image))

        return [p for p in parts if p and p.strip()]

    def _handle_table(self, block: LayoutBlock) -> str:
        content = block.content
        if "<table" in content.lower() or "<td" in content.lower():
            md = self.analyzer.html_table_to_markdown(content)
            if md and "|" in md:
                return md
            return ""
        if "|" in content and "---" in content:
            return content.strip()
        return ""

    def _handle_figure(self, block: LayoutBlock, page_image: "Image.Image") -> str:
        # Caption nếu PPStructureV3 đã gán label figure_caption riêng,
        # dùng luôn content đó làm phần mô tả; nếu không, fallback generic.
        caption = block.content.strip()
        label = caption if caption and not caption.startswith("[FIGURE") else "Hình minh họa"
        return f"[FIGURE: {label}]"

    def _handle_text(self, block: LayoutBlock, page_image: "Image.Image") -> str:
        text = block.content

        # Handwriting-aware retry: nếu block có nhiều dòng confidence thấp,
        # và có bbox để crop lại, thử OCR lại bằng ảnh upscale.
        if (block.scores and block.bbox
                and block.low_confidence_ratio >= HANDWRITING_LOW_CONF_RATIO):
            retried = self._retry_low_confidence_block(block, page_image)
            if retried and len(retried.strip()) >= len(text.strip()) * 0.5:
                logger.debug(
                    f"[Handwriting] Retried block (low_conf_ratio="
                    f"{block.low_confidence_ratio:.0%}), using upscaled OCR result"
                )
                text = retried

        return self.corrector.correct(text)

    def _retry_low_confidence_block(self, block: LayoutBlock, page_image: "Image.Image") -> Optional[str]:
        try:
            bbox = block.bbox
            if not bbox or len(bbox) < 4:
                return None
            x0, y0, x1, y1 = bbox[:4]
            crop = page_image.crop((int(x0), int(y0), int(x1), int(y1)))
            return self.ocr_engine.extract_text_upscaled(crop)
        except Exception as e:
            logger.debug(f"Low-confidence retry failed: {e}")
            return None


# ============================================================================
# HYBRID PDF PROCESSOR — PRODUCTION v6
# ============================================================================

class HybridPDFProcessor:
    """
    TEXT  → pdfplumber fast-path (không render, không OCR — đã tối ưu đúng
            ở v5.1, giữ nguyên).
    TABLE/IMAGE/MIXED → render ảnh → LayoutAnalyzer.analyze() MỘT LẦN →
            LayoutRouter route block-by-block → fuse theo reading order.

    Page separator: "\n\n---\n\n" giữa các trang (cho SmartChunker).
    """

    def __init__(
        self,
        ocr_engine: PaddleOCREngine,
        analyzer: LayoutAnalyzer,
        corrector: VietnameseTextCorrector,
        image_scale: float = 1.5,
    ):
        self.ocr_engine = ocr_engine
        self.analyzer = analyzer
        self.corrector = corrector
        self.image_scale = image_scale
        self.classifier = PageClassifier()
        self.img_extractor = PDFImageExtractor()
        self.router = LayoutRouter(corrector=corrector, ocr_engine=ocr_engine, analyzer=analyzer)

    def process(self, file_path: str) -> Optional[str]:
        try:
            import pdfplumber
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ImportError(f"Missing: {e}. Run: pip install pdfplumber pdf2image") from e

        t0 = time.time()
        with pdfplumber.open(file_path) as pdf:
            total = len(pdf.pages)
            logger.info(f"PDF: {total} pages | {Path(file_path).name}")

            page_types = [self._classify(p) for p in pdf.pages]
            type_counts = {t: page_types.count(t) for t in set(page_types)}
            logger.info(f"Page types: {type_counts}")

            render_needed = [i for i, t in enumerate(page_types) if t != PageType.TEXT]
            rendered: Dict[int, Image.Image] = {}
            if render_needed:
                logger.info(f"Rendering {len(render_needed)} non-text pages (DPI={PDF_RENDER_DPI})...")
                all_pages = convert_from_path(file_path, dpi=PDF_RENDER_DPI, fmt="RGB")
                for idx in render_needed:
                    if idx < len(all_pages):
                        rendered[idx] = all_pages[idx]

            page_results: List[str] = []
            for page_num, (plumber_page, ptype) in enumerate(zip(pdf.pages, page_types), 1):
                logger.info(f"  [{page_num:3d}/{total}] type={ptype}")
                content = self._process_page(
                    plumber_page=plumber_page,
                    page_num=page_num,
                    page_type=ptype,
                    rendered_image=rendered.get(page_num - 1),
                )
                if content.strip():
                    page_results.append(content)

        merged = "\n\n---\n\n".join(page_results)
        merged = self._post_process(merged)

        elapsed = time.time() - t0
        logger.info(
            f"✅ PDF done: {total} pages in {elapsed:.1f}s "
            f"({elapsed/total:.1f}s/page) | {len(merged)} chars"
        )
        return merged

    def _classify(self, page) -> str:
        try:
            return self.classifier.classify(page, page.width, page.height)
        except Exception:
            return PageType.MIXED

    def _process_page(
        self,
        plumber_page,
        page_num: int,
        page_type: str,
        rendered_image: Optional[Image.Image],
    ) -> str:
        # ── TEXT: fast-path không render, không OCR ──────────────────────
        if page_type == PageType.TEXT:
            text = PlumberExtractor.extract_page(plumber_page, page_num)
            parts = [self.corrector.correct(text)] if text.strip() else []
            parts.extend(self.img_extractor.extract_images_from_page(plumber_page, page_num))
            return "\n\n".join(p for p in parts if p.strip())

        # ── TABLE / IMAGE / MIXED: layout pipeline thống nhất ────────────
        if rendered_image is not None and self.analyzer.available:
            parts = self.router.route_page(rendered_image)
            if parts:
                # Vẫn lấy thêm embedded images không nằm trong figure block
                # (vd. logo/watermark mà layout model bỏ qua) để không mất
                # dữ liệu — nhưng không lặp lại text vì router đã xử lý.
                return "\n\n".join(parts)

        # ── Fallback nếu layout analyzer không khả dụng ──────────────────
        if rendered_image is not None:
            text = self.ocr_engine.extract_text(rendered_image)
            if text.strip():
                return self.corrector.correct(text)

        text = PlumberExtractor.extract_page(plumber_page, page_num)
        return self.corrector.correct(text) if text.strip() else ""

    # ── Post-processing (giữ nguyên logic v5.1) ──────────────────────────

    def _post_process(self, text: str) -> str:
        if not text:
            return ""

        _IMG_RE = re.compile(
            r'!\[[^\]]*\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)', re.DOTALL
        )
        image_blocks: List[str] = []

        def extract_img(m: re.Match) -> str:
            idx = len(image_blocks)
            image_blocks.append(m.group(0))
            return f"%%IMG_{idx:04d}%%"

        text = _IMG_RE.sub(extract_img, text)
        text = self._basic_cleanup(text)
        text = self._normalize_tables(text)
        text = self._fix_headings(text)
        text = self._rule_based_corrections(text)

        for idx, block in enumerate(image_blocks):
            text = text.replace(f"%%IMG_{idx:04d}%%", block)

        return text.strip()

    @staticmethod
    def _basic_cleanup(text: str) -> str:
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        text = re.sub(r'[ \t]+\n', '\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'(?m)^(Trang|Page)\s+\d+\s*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?m)^\d+\s*$', '', text)
        lines = text.split('\n')
        counts: Dict[str, int] = {}
        for ln in lines:
            s = ln.strip()
            if s and len(s) < 60:
                counts[s] = counts.get(s, 0) + 1
        text = '\n'.join(ln for ln in lines if counts.get(ln.strip(), 0) <= 3)
        return text

    @staticmethod
    def _normalize_tables(text: str) -> str:
        _TABLE_BLOCK_RE = re.compile(r'(?m)((?:^\|.+\|\s*\n?)+)')

        def normalize_block(m: re.Match) -> str:
            lines = [ln.strip() for ln in m.group(0).strip().split('\n') if ln.strip()]
            rows = []
            for ln in lines:
                if re.match(r'^\|[-:\s|]+\|$', ln):
                    continue
                if ln.startswith('|'):
                    rows.append([c.strip() for c in ln.split('|')[1:-1]])
            if not rows:
                return m.group(0)
            max_cols = max(len(r) for r in rows)
            norm = [r + [''] * (max_cols - len(r)) for r in rows]
            table_lines = ['| ' + ' | '.join(norm[0]) + ' |']
            table_lines.append('| ' + ' | '.join(['---'] * max_cols) + ' |')
            for row in norm[1:]:
                table_lines.append('| ' + ' | '.join(c if c else '-' for c in row) + ' |')
            return '\n'.join(table_lines)

        return _TABLE_BLOCK_RE.sub(normalize_block, text)

    @staticmethod
    def _fix_headings(text: str) -> str:
        lines = text.split('\n')
        result: List[str] = []
        for ln in lines:
            s = ln.strip()
            if (s and len(s) < 80 and s.isupper()
                    and not s.startswith('#')
                    and not s.startswith('|')
                    and not s.startswith('!')
                    and len(s.split()) > 1):
                ln = f"## {s}"
                s = ln
            if s.startswith('#') and result and result[-1].strip():
                result.append('')
            result.append(ln)
        return '\n'.join(result)

    @staticmethod
    def _rule_based_corrections(text: str) -> str:
        corrections = {
            r'\bViet Nam\b': 'Việt Nam',
            r'\bHa Noi\b': 'Hà Nội',
            r'\bHo Chi Minh\b': 'Hồ Chí Minh',
            r'(\d)\s+\.(\d)': r'\1.\2',
            r'(\d)\s+,(\d)': r'\1,\2',
            r'\bNgan hang\b': 'Ngân hàng',
            r'\bchinh sach\b': 'chính sách',
            r'\bNHCSXH\b': 'NHCSXH',
        }
        for pat, repl in corrections.items():
            try:
                text = re.sub(pat, repl, text)
            except Exception:
                pass
        return text


# ============================================================================
# DOCX / EXCEL HANDLERS  (không đổi so với v5.1)
# ============================================================================

class DocxProcessor:
    @staticmethod
    def process(file_path: str, corrector: VietnameseTextCorrector) -> Optional[str]:
        try:
            import docx
            from docx import Document
            doc = Document(file_path)
            sections: List[str] = []
            for element in doc.element.body:
                tag = element.tag.split('}')[-1] if '}' in element.tag else element.tag
                if tag == 'p':
                    para = docx.text.paragraph.Paragraph(element, doc)
                    text = para.text.strip()
                    if not text:
                        continue
                    style = para.style.name.lower()
                    if 'heading 1' in style:
                        sections.append(f"# {text}")
                    elif 'heading 2' in style:
                        sections.append(f"## {text}")
                    elif 'heading 3' in style:
                        sections.append(f"### {text}")
                    else:
                        sections.append(corrector.correct(text))
                elif tag == 'tbl':
                    table = docx.table.Table(element, doc)
                    rows = [
                        [cell.text.strip().replace('\n', ' ') for cell in row.cells]
                        for row in table.rows
                    ]
                    md = PlumberExtractor._table_to_markdown(rows)
                    if md:
                        sections.append(md)
            return "\n\n".join(sections)
        except Exception as e:
            logger.error(f"DOCX processing failed: {e}", exc_info=True)
            return DocxProcessor._convert_via_libreoffice(file_path)

    @staticmethod
    def _convert_via_libreoffice(file_path: str) -> Optional[str]:
        try:
            import subprocess
            with tempfile.TemporaryDirectory() as tmpdir:
                subprocess.run(
                    ['libreoffice', '--headless', '--convert-to', 'pdf',
                     '--outdir', tmpdir, file_path],
                    capture_output=True, text=True, timeout=60,
                )
                pdf_files = list(Path(tmpdir).glob('*.pdf'))
                if pdf_files:
                    return None
        except Exception as e:
            logger.error(f"LibreOffice conversion failed: {e}")
        return None


class ExcelProcessor:
    @staticmethod
    def process(file_path: str) -> Optional[str]:
        try:
            import pandas as pd
            excel_file = pd.ExcelFile(file_path)
            sections: List[str] = []
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                if df.empty:
                    continue
                sections.append(f"## {sheet_name}")
                header_row = ExcelProcessor._detect_header_row(df)
                if header_row >= 0:
                    df.columns = df.iloc[header_row].astype(str)
                    df = df.iloc[header_row + 1:].reset_index(drop=True)
                sections.append(ExcelProcessor._dataframe_to_markdown(df))
            return "\n\n".join(sections)
        except Exception as e:
            logger.error(f"Excel processing failed: {e}")
            return None

    @staticmethod
    def _detect_header_row(df) -> int:
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            if sum(1 for v in row if isinstance(v, str) and v.strip()) / len(row) > 0.6:
                return i
        return 0

    @staticmethod
    def _dataframe_to_markdown(df) -> str:
        import pandas as pd
        rows = [[str(c) for c in df.columns]]
        for _, row in df.iterrows():
            rows.append(['-' if pd.isna(v) else str(v).strip() for v in row])
        return PlumberExtractor._table_to_markdown(rows)


# ============================================================================
# PUBLIC API: PaddleOCRVLProcessor (backward compatible — KHÔNG đổi tên/sig)
# ============================================================================

class PaddleOCRVLProcessor:
    """
    Public API — interface tương thích với document_processor.py.
    Nội bộ v6 dùng LayoutAnalyzer (PPStructureV3) làm nguồn duy nhất cho
    layout + table + figure + text trên page cần render.
    """

    def __init__(
        self,
        use_gpu: bool = True,
        image_scale: float = 1.5,
        max_new_tokens: int = 800,          # kept for compat
        enable_spell_correction: bool = True,
        enable_table_normalization: bool = True,
        batch_size: int = 4,                # kept for compat
        use_flash_attention: bool = True,   # kept for compat
        use_compile: bool = False,          # kept for compat
        gpu_id: int = 0,
        correction_model: str = "teacher",
        lang: str = "vi",
    ):
        self.use_gpu = use_gpu
        self.image_scale = image_scale
        self.lang = lang

        logger.info("🚀 Initializing PaddleOCR v3.7.0 processor (v6 layout-first)...")

        self._corrector = VietnameseTextCorrector(
            model_size=correction_model,
            enabled=enable_spell_correction,
        )
        self._ocr_engine = PaddleOCREngine(use_gpu=use_gpu, lang=lang)
        self._analyzer = LayoutAnalyzer(use_gpu=use_gpu, lang=lang)
        self._pdf_proc = HybridPDFProcessor(
            ocr_engine=self._ocr_engine,
            analyzer=self._analyzer,
            corrector=self._corrector,
            image_scale=image_scale,
        )

        logger.info("✅ PaddleOCRVLProcessor v6 ready")

    def process_pdf(self, file_path: str) -> Optional[str]:
        logger.info(f"📄 process_pdf: {Path(file_path).name}")
        result = self._pdf_proc.process(file_path)
        if not result or len(result.strip()) < 50:
            raise ValueError(f"PDF processing returned insufficient content: {file_path}")
        return result

    def process_docx(self, file_path: str) -> Optional[str]:
        logger.info(f"📝 process_docx: {Path(file_path).name}")
        result = DocxProcessor.process(file_path, self._corrector)
        if not result:
            try:
                import subprocess
                with tempfile.TemporaryDirectory() as tmpdir:
                    subprocess.run(
                        ['libreoffice', '--headless', '--convert-to', 'pdf',
                         '--outdir', tmpdir, file_path],
                        capture_output=True, text=True, timeout=60,
                    )
                    pdf_files = list(Path(tmpdir).glob('*.pdf'))
                    if pdf_files:
                        return self.process_pdf(str(pdf_files[0]))
            except Exception as e:
                logger.error(f"LibreOffice fallback failed: {e}")
        return result

    def process_image(self, file_path: str) -> Optional[str]:
        logger.info(f"🖼️  process_image: {Path(file_path).name}")
        try:
            image = Image.open(file_path).convert("RGB")
            if self._analyzer.available:
                router = LayoutRouter(
                    corrector=self._corrector,
                    ocr_engine=self._ocr_engine,
                    analyzer=self._analyzer,
                )
                parts = router.route_page(image)
                if parts:
                    return self._post_process("\n\n".join(parts))

            text = self._ocr_engine.extract_text(image)
            if text:
                corrected = self._corrector.correct(text)
                return self._post_process(corrected)
            return None
        except Exception as e:
            logger.error(f"Image processing failed: {e}", exc_info=True)
            return None

    def _process_excel(self, file_path: str) -> Optional[str]:
        logger.info(f"📊 process_excel: {Path(file_path).name}")
        result = ExcelProcessor.process(file_path)
        if result:
            return self._post_process(result)
        return None

    def _post_process(self, text: str) -> str:
        return self._pdf_proc._post_process(text)


# ============================================================================
# GLOBAL SINGLETON
# ============================================================================

_instance: Optional[PaddleOCRVLProcessor] = None
_lock = threading.Lock()


def get_paddle_ocr_processor(
    use_gpu: bool = True,
    image_scale: float = 1.5,
    max_new_tokens: int = 800,
    enable_spell_correction: bool = True,
    enable_table_normalization: bool = True,
    batch_size: int = 4,
    use_flash_attention: bool = True,
    use_compile: bool = False,
    gpu_id: int = 0,
    correction_model: str = "teacher",
    lang: str = "vi",
) -> PaddleOCRVLProcessor:
    """Return global singleton (thread-safe)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = PaddleOCRVLProcessor(
                    use_gpu=use_gpu,
                    image_scale=image_scale,
                    max_new_tokens=max_new_tokens,
                    enable_spell_correction=enable_spell_correction,
                    enable_table_normalization=enable_table_normalization,
                    batch_size=batch_size,
                    use_flash_attention=use_flash_attention,
                    use_compile=use_compile,
                    gpu_id=gpu_id,
                    correction_model=correction_model,
                    lang=lang,
                )
    return _instance