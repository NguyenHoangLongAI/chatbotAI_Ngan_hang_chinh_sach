"""
PaddleOCR v3.7.0 Document Processor — PRODUCTION v5
=====================================================
Viết lại hoàn toàn từ v4, fix tất cả vấn đề production:

ROOT CAUSE FIXES:
  1. PPStructureV3 detect text thường thành bảng giả
     → Chỉ dùng PPStructureV3 khi page thực sự có bảng (TABLE type)
     → MIXED/TEXT pages dùng PaddleOCR plain text
  2. pages_detected = 1 → page separator mất
     → HybridPDFProcessor.process() giữ nguyên "\n\n---\n\n"
  3. ProtonX không chạy → text lỗi
     → Corrector apply đúng chỗ, không bỏ qua
  4. Table duplicate (text + table của cùng content)
     → Bỏ "bổ sung bảng từ PPStructureV3 nếu pdfplumber đã có text"
  5. PaddleOCR v3.7.0 API: use_gpu → device, show_log removed

CHIẾN LƯỢC XỬ LÝ PAGE:
  TEXT  → pdfplumber (nhanh, không cần render)
  TABLE → pdfplumber text + PPStructureV3 chỉ lấy bảng HTML
  IMAGE → PPStructureV3 full layout → PaddleOCR fallback
  MIXED → pdfplumber nếu đủ text (≥80 chars), else PaddleOCR

OCR ENGINE:
  PaddleOCR(text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec")
  PPStructureV3(same models, use_seal/formula/chart=False)

TEXT CORRECTION:
  ProtonX student model, offline mode, chỉ correct plain text paragraphs
  Bỏ qua markdown table, code block, base64 image
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# ── Suppress PaddleX verbose "Creating model" logs ──────────────────
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
logging.getLogger("paddlex").setLevel(logging.WARNING)
logging.getLogger("paddleocr").setLevel(logging.WARNING)
logging.getLogger("paddle").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

TEXT_PAGE_MIN_CHARS = 80   # chars để classify page là TEXT (pdfplumber fast-path)
MIXED_PAGE_MIN_CHARS = 40  # chars để dùng pdfplumber thay vì OCR trong MIXED pages
PDF_RENDER_DPI = 150


# ============================================================================
# PROTONX VIETNAMESE TEXT CORRECTOR
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

    def __init__(self, model_size: str = "student", enabled: bool = True):
        self.enabled = enabled
        self.model_size = model_size
        self._client = None
        self._lock = threading.Lock()
        if enabled:
            self._init_client()

    def _init_client(self):
        try:
            # Force offline — không check HuggingFace mỗi request
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
            return text
        try:
            return self._correct_with_preservation(text)
        except Exception as e:
            logger.debug(f"Text correction failed: {e}")
            return text

    def _correct_with_preservation(self, text: str) -> str:
        """Preserve markdown tables, base64 images, code blocks; correct plain text."""
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
        try:
            model_name = self.MODELS.get(self.model_size, self.MODELS["student"])
            result = self._client.text.correct(input=text, top_k=1, model=model_name)
            if (result and "data" in result and result["data"]
                    and result["data"][0].get("candidates")):
                return result["data"][0]["candidates"][0]["output"]
            return text
        except Exception as e:
            logger.debug(f"ProtonX API error: {e}")
            return text


# ============================================================================
# PAGE TYPE
# ============================================================================

class PageType:
    TEXT  = "text"
    TABLE = "table"
    IMAGE = "image"
    MIXED = "mixed"


# ============================================================================
# PAGE CLASSIFIER
# ============================================================================

class PageClassifier:
    """
    Phân loại page dựa trên pdfplumber metadata.
    Không dùng OpenCV.
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

        # TEXT: nhiều text, ít ảnh → pdfplumber fast-path
        if char_count >= TEXT_PAGE_MIN_CHARS and image_coverage < 0.30:
            return PageType.TABLE if has_table else PageType.TEXT

        # IMAGE: ảnh chiếm nhiều, rất ít text
        if image_coverage >= 0.40 and char_count < 30:
            return PageType.IMAGE

        # TABLE: pdfplumber detect được bảng
        if has_table:
            return PageType.TABLE

        # MIXED: còn lại — có cả text lẫn ảnh, hoặc text ít
        # Nếu có đủ text dùng được → vẫn xử lý như TEXT để tránh chậm
        if char_count >= MIXED_PAGE_MIN_CHARS:
            return PageType.TEXT

        return PageType.MIXED


# ============================================================================
# PDF EMBEDDED IMAGE EXTRACTOR
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
                    label = f"Page {page_num} Image {idx + 1}"
                    markdown_images.append(f"![{label}](data:{mime};base64,{b64})")
            except Exception as e:
                logger.debug(f"Image extract error p{page_num} img{idx}: {e}")
        return markdown_images

    @staticmethod
    def _detect_mime(data: bytes) -> str:
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        if data[:3] == b'\xff\xd8\xff':
            return "image/jpeg"
        return "image/png"


# ============================================================================
# PDFPLUMBER FAST-PATH EXTRACTOR
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
# PADDLEOCR v3.7.0 ENGINE  (PP-OCRv5_mobile)
# ============================================================================

class PaddleOCREngine:
    """
    PaddleOCR v3.7.0 với PP-OCRv5_mobile models.
    PP-OCRv6 bị lỗi PIR strides với PaddlePaddle 3.0.0 → force PP-OCRv5_mobile.
    """

    def __init__(self, use_gpu: bool = True, lang: str = "vi"):
        self._ocr = None
        self.use_gpu = use_gpu
        self.lang = lang
        self._init()

    def _init(self):
        try:
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
            )
            logger.info("✅ PaddleOCR v3.7.0 (PP-OCRv5_mobile) engine ready")
        except ImportError:
            logger.warning("⚠️ paddleocr not installed. Run: pip install paddleocr==3.7.0")
        except Exception as e:
            logger.warning(f"⚠️ PaddleOCR init failed: {e}")

    def extract_text(self, image: "Image.Image") -> str:
        """Extract plain text từ PIL Image dùng predict() API."""
        if self._ocr is None:
            return ""
        try:
            import numpy as np
            img_array = np.array(image.convert("RGB"))
            result = self._ocr.predict(img_array)
            # Output: list of page dicts với keys: rec_texts, rec_scores
            lines = []
            for page in result:
                rec_texts  = page.get("rec_texts", []) or []
                rec_scores = page.get("rec_scores", []) or []
                for i, text in enumerate(rec_texts):
                    score = rec_scores[i] if i < len(rec_scores) else 1.0
                    if text and str(text).strip() and score > 0.3:
                        lines.append(str(text).strip())
            return "\n".join(lines)
        except Exception as e:
            logger.error(f"PaddleOCR extraction error: {e}")
            return ""


# ============================================================================
# PP-STRUCTURE V3 TABLE EXTRACTOR
# ============================================================================

class PPStructureTableExtractor:
    """
    PPStructureV3 chỉ dùng để extract BẢNG từ page ảnh.
    KHÔNG dùng cho text extraction thông thường (tránh detect text → table sai).

    Key insight từ thực tế:
    - parsing_res_list[i].label == "table" → content là HTML table
    - parsing_res_list[i].label == "text"/"paragraph" → text thường, BỎ QUA
    - Chỉ lấy label == "table", convert HTML → Markdown
    """

    def __init__(self, use_gpu: bool = True, lang: str = "vi"):
        self._pipeline = None
        self.use_gpu = use_gpu
        self.lang = lang
        self._init()

    def _init(self):
        try:
            os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
            from paddleocr import PPStructureV3
            self._pipeline = PPStructureV3(
                lang=self.lang,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                use_doc_orientation_classify=False,
                use_seal_recognition=False,
                use_formula_recognition=False,
                use_chart_recognition=False,
            )
            logger.info("✅ PP-StructureV3 table extractor ready (PP-OCRv5_mobile)")
        except ImportError:
            logger.warning(
                "⚠️ paddleocr/paddlex[ocr] not installed — "
                "table extraction disabled. Run: pip install paddlex[ocr]"
            )
        except Exception as e:
            logger.warning(f"⚠️ PP-StructureV3 init failed: {e}")

    def extract_tables_only(self, image: "Image.Image") -> List[str]:
        """
        Extract CHỈ BẢNG từ image, trả về list Markdown table strings.
        Bỏ qua tất cả text blocks để tránh duplicate với pdfplumber.
        """
        if self._pipeline is None:
            return []
        try:
            import numpy as np
            img_array = np.array(image.convert("RGB"))
            result = self._pipeline.predict(img_array)

            tables: List[str] = []
            for page_result in result:
                parsing_list = page_result.get("parsing_res_list", [])
                for block in parsing_list:
                    label = (getattr(block, "label", "") or "").lower()
                    if label != "table":
                        continue
                    content = getattr(block, "content", "") or ""
                    if not content.strip():
                        continue
                    # content là HTML table từ PPStructureV3
                    if "<table" in content.lower() or "<td" in content.lower():
                        md = self._html_to_markdown(content)
                        if md and "|" in md:
                            tables.append(md)
                    # Hoặc đã là Markdown
                    elif "|" in content and "---" in content:
                        tables.append(content.strip())

            return tables
        except Exception as e:
            logger.error(f"PP-StructureV3 table extraction error: {e}")
            return []

    def extract_full_layout(self, image: "Image.Image") -> List[Dict]:
        """
        Full layout extraction cho IMAGE pages.
        Trả về list dicts: {type, content}
        type: "table" | "text" | "figure" | "formula"
        """
        if self._pipeline is None:
            return []
        try:
            import numpy as np
            img_array = np.array(image.convert("RGB"))
            result = self._pipeline.predict(img_array)

            elements: List[Dict] = []
            for page_result in result:
                parsing_list = page_result.get("parsing_res_list", [])
                for block in parsing_list:
                    label   = (getattr(block, "label", "") or "").lower()
                    content = (getattr(block, "content", "") or "").strip()
                    if not content:
                        continue

                    if label == "table":
                        if "<table" in content.lower() or "<td" in content.lower():
                            md = self._html_to_markdown(content)
                            if md:
                                elements.append({"type": "table", "content": md})
                        elif "|" in content:
                            elements.append({"type": "table", "content": content})
                    elif label in ("figure", "image", "figure_caption"):
                        elements.append({"type": "figure", "content": f"[FIGURE: {content}]"})
                    elif label == "formula":
                        elements.append({"type": "formula", "content": f"$${content}$$"})
                    else:
                        # text, paragraph, title, list → plain text
                        elements.append({"type": "text", "content": content})

                # Fallback nếu parsing_res_list rỗng
                if not elements:
                    ocr_res = page_result.get("overall_ocr_res", {}) or {}
                    rec_texts = ocr_res.get("rec_texts", []) or []
                    if rec_texts:
                        elements.append({
                            "type": "text",
                            "content": "\n".join(str(t) for t in rec_texts if t),
                        })

            return elements
        except Exception as e:
            logger.error(f"PP-StructureV3 full layout error: {e}")
            return []

    @staticmethod
    def _html_to_markdown(html: str) -> str:
        """Convert HTML table → Markdown table."""
        try:
            from html.parser import HTMLParser

            class TableParser(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.rows: List[List[str]] = []
                    self._row: List[str] = []
                    self._cell: str = ""
                    self._in_cell = False

                def handle_starttag(self, tag, attrs):
                    if tag == "tr":
                        self._row = []
                    elif tag in ("td", "th"):
                        self._cell = ""
                        self._in_cell = True

                def handle_endtag(self, tag):
                    if tag in ("td", "th"):
                        self._row.append(self._cell.strip())
                        self._in_cell = False
                    elif tag == "tr":
                        if self._row:
                            self.rows.append(self._row)

                def handle_data(self, data):
                    if self._in_cell:
                        self._cell += data

            parser = TableParser()
            parser.feed(html)
            if not parser.rows:
                return ""

            rows = parser.rows
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


# ============================================================================
# HYBRID PDF PROCESSOR — PRODUCTION v5
# ============================================================================

class HybridPDFProcessor:
    """
    Chiến lược xử lý PDF theo page type:

    TEXT  → pdfplumber fast-path (không OCR, không render)
    TABLE → pdfplumber text + PPStructureV3 extract tables only
    IMAGE → PPStructureV3 full layout → PaddleOCR fallback
    MIXED → pdfplumber nếu ≥ MIXED_PAGE_MIN_CHARS, else PaddleOCR

    Page separator: "\n\n---\n\n" giữa các trang (cho SmartChunker phân trang).
    ProtonX correction: apply cho tất cả text output.
    """

    def __init__(
        self,
        ocr_engine: PaddleOCREngine,
        table_extractor: PPStructureTableExtractor,
        corrector: VietnameseTextCorrector,
        image_scale: float = 1.5,
    ):
        self.ocr_engine = ocr_engine
        self.table_extractor = table_extractor
        self.corrector = corrector
        self.image_scale = image_scale
        self.classifier = PageClassifier()
        self.img_extractor = PDFImageExtractor()

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

            # Chỉ render các page không phải TEXT
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
                    total=total,
                )
                if content.strip():
                    page_results.append(content)

        # Giữ page separator để SmartChunker phân trang đúng
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
        total: int,
    ) -> str:
        parts: List[str] = []
        image_blocks = self.img_extractor.extract_images_from_page(plumber_page, page_num)

        if page_type == PageType.TEXT:
            # ── Fast path: pdfplumber, không OCR ─────────────────────
            text = PlumberExtractor.extract_page(plumber_page, page_num)
            if text.strip():
                parts.append(self.corrector.correct(text))

        elif page_type == PageType.TABLE:
            # ── pdfplumber text + PPStructureV3 chỉ lấy bảng ─────────
            plumber_text = PlumberExtractor.extract_page(plumber_page, page_num)
            if plumber_text.strip():
                parts.append(self.corrector.correct(plumber_text))

            # PPStructureV3: extract thêm bảng mà pdfplumber bỏ sót
            # Chỉ lấy table blocks, KHÔNG lấy text blocks (tránh duplicate)
            if rendered_image is not None and self.table_extractor._pipeline is not None:
                extra_tables = self.table_extractor.extract_tables_only(rendered_image)
                for tbl in extra_tables:
                    # Kiểm tra không duplicate với pdfplumber tables
                    # (so sánh đơn giản bằng độ dài content)
                    if tbl not in plumber_text:
                        parts.append(tbl)

        elif page_type == PageType.IMAGE:
            # ── Full OCR: PPStructureV3 layout → PaddleOCR fallback ──
            if rendered_image is not None:
                if self.table_extractor._pipeline is not None:
                    elements = self.table_extractor.extract_full_layout(rendered_image)
                    for el in elements:
                        c = el.get("content", "").strip()
                        if not c:
                            continue
                        if el["type"] == "table":
                            parts.append(c)
                        elif el["type"] == "text":
                            parts.append(self.corrector.correct(c))
                        else:
                            parts.append(c)

                if not parts:
                    # Fallback: PaddleOCR plain text
                    text = self.ocr_engine.extract_text(rendered_image)
                    if text.strip():
                        parts.append(self.corrector.correct(text))
            else:
                text = PlumberExtractor.extract_page(plumber_page, page_num)
                if text.strip():
                    parts.append(self.corrector.correct(text))

        else:  # MIXED
            # ── pdfplumber nếu đủ text, else PaddleOCR ───────────────
            plumber_text = PlumberExtractor.extract_page(plumber_page, page_num)
            plumber_chars = len(plumber_text.strip())

            if plumber_chars >= MIXED_PAGE_MIN_CHARS:
                # Đủ text từ pdfplumber → dùng luôn (nhanh hơn OCR)
                parts.append(self.corrector.correct(plumber_text))
                # Bổ sung bảng nếu PPStructureV3 available
                if rendered_image is not None and self.table_extractor._pipeline is not None:
                    extra_tables = self.table_extractor.extract_tables_only(rendered_image)
                    for tbl in extra_tables:
                        if tbl not in plumber_text:
                            parts.append(tbl)
            else:
                # Quá ít text → OCR
                if rendered_image is not None:
                    if self.table_extractor._pipeline is not None:
                        elements = self.table_extractor.extract_full_layout(rendered_image)
                        for el in elements:
                            c = el.get("content", "").strip()
                            if not c:
                                continue
                            if el["type"] == "table":
                                parts.append(c)
                            elif el["type"] == "text":
                                parts.append(self.corrector.correct(c))
                            else:
                                parts.append(c)

                    if not parts:
                        text = self.ocr_engine.extract_text(rendered_image)
                        if text.strip():
                            parts.append(self.corrector.correct(text))
                elif plumber_text.strip():
                    parts.append(self.corrector.correct(plumber_text))

        parts.extend(image_blocks)
        return "\n\n".join(p for p in parts if p.strip())

    # ── Post-processing ──────────────────────────────────────────────

    def _post_process(self, text: str) -> str:
        if not text:
            return ""

        # Protect base64 images
        _IMG_RE = re.compile(
            r'!\[[^\]]*\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)', re.DOTALL
        )
        image_blocks: List[str] = []

        def extract_img(m: re.Match) -> str:
            idx = len(image_blocks)
            image_blocks.append(m.group(0))
            return f"%%IMG_{idx:04d}%%"

        # Protect page separators
        SEP = "\n\n---\n\n"
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
        # Bỏ page number artifacts nhưng giữ separator
        text = re.sub(r'(?m)^(Trang|Page)\s+\d+\s*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?m)^\d+\s*$', '', text)
        # Bỏ dòng lặp lại (header/footer)
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
            r'\bViet Nam\b':    'Việt Nam',
            r'\bHa Noi\b':      'Hà Nội',
            r'\bHo Chi Minh\b': 'Hồ Chí Minh',
            r'(\d)\s+\.(\d)':   r'\1.\2',
            r'(\d)\s+,(\d)':    r'\1,\2',
            r'\bNgan hang\b':   'Ngân hàng',
            r'\bchinh sach\b':  'chính sách',
            r'\bNHCSXH\b':      'NHCSXH',
        }
        for pat, repl in corrections.items():
            try:
                text = re.sub(pat, repl, text)
            except Exception:
                pass
        return text


# ============================================================================
# DOCX / EXCEL HANDLERS
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
                    if   'heading 1' in style: sections.append(f"# {text}")
                    elif 'heading 2' in style: sections.append(f"## {text}")
                    elif 'heading 3' in style: sections.append(f"### {text}")
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
        """Fallback: convert DOCX → PDF via LibreOffice rồi OCR."""
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
                    # Note: caller phải cung cấp processor để gọi process_pdf
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
        import pandas as pd
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
# PUBLIC API: PaddleOCRVLProcessor (backward compatible)
# ============================================================================

class PaddleOCRVLProcessor:
    """
    Public API — interface tương thích với document_processor.py.
    Nội bộ dùng PaddleOCR v3.7.0 + PPStructureV3 + ProtonX correction.
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
        correction_model: str = "student",
        lang: str = "vi",
    ):
        self.use_gpu = use_gpu
        self.image_scale = image_scale
        self.lang = lang

        logger.info("🚀 Initializing PaddleOCR v3.7.0 processor...")

        self._corrector = VietnameseTextCorrector(
            model_size=correction_model,
            enabled=enable_spell_correction,
        )
        self._ocr_engine = PaddleOCREngine(use_gpu=use_gpu, lang=lang)
        self._table_extractor = PPStructureTableExtractor(use_gpu=use_gpu, lang=lang)
        self._pdf_proc = HybridPDFProcessor(
            ocr_engine=self._ocr_engine,
            table_extractor=self._table_extractor,
            corrector=self._corrector,
            image_scale=image_scale,
        )

        logger.info("✅ PaddleOCRVLProcessor v5 ready")

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
            # Fallback: convert via LibreOffice
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
            # Thử PPStructureV3 full layout trước
            if self._table_extractor._pipeline is not None:
                elements = self._table_extractor.extract_full_layout(image)
                if elements:
                    parts = []
                    for el in elements:
                        c = el.get("content", "").strip()
                        if not c:
                            continue
                        if el["type"] == "table":
                            parts.append(c)
                        elif el["type"] == "text":
                            parts.append(self._corrector.correct(c))
                        else:
                            parts.append(c)
                    if parts:
                        return self._post_process("\n\n".join(parts))

            # Fallback: PaddleOCR plain text
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
    correction_model: str = "student",
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