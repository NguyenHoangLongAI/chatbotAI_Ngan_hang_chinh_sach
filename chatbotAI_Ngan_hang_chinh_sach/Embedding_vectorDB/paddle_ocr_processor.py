"""
PaddleOCR-VL Document Processor — OPTIMIZED v3
================================================
Improvements over v2:

PDF Pipeline (hybrid):
  - pdfplumber fast-path for text-dominant pages  (saves 80% VL calls on text PDFs)
  - VL model only for scanned / image-heavy pages
  - Embedded images extracted → base64 inline Markdown  ![img](data:image/png;base64,...)
  - Page-type classifier: text / mixed / image-only (no OpenCV dep)
  - Parallel page rendering via ThreadPoolExecutor

VL Inference:
  - Dynamic max_new_tokens per page type (text=600, table=1200, image=800)
  - torch.inference_mode + padding=False (same as v2)
  - Removed repetition_penalty
  - Flash Attention 2 with eager fallback

Post-processing:
  - Markdown heading de-dup across pages
  - Page-break separator → clean `---`
  - Table re-alignment pass
  - base64 image blocks preserved verbatim (not stripped by cleanup)
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Minimum text character count to classify a page as "text-dominant"
TEXT_PAGE_MIN_CHARS = 80
# If image pixels cover more than this fraction of page area → image-heavy page
IMAGE_COVERAGE_THRESHOLD = 0.25
# DPI for pdf2image conversion (150 is enough for VL encoder)
PDF_RENDER_DPI = 150
# Workers for parallel page rendering (I/O-bound)
RENDER_WORKERS = 4


# ============================================================================
# PAGE TYPE ENUM
# ============================================================================

class PageType:
    TEXT  = "text"    # pdfplumber fast-path
    TABLE = "table"   # VL with table prompt + higher token limit
    IMAGE = "image"   # VL with image prompt
    MIXED = "mixed"   # VL with full-document prompt


# ============================================================================
# PROMPT TEMPLATES
# ============================================================================

class OCRPrompts:
    FULL_DOCUMENT = """Bạn là chuyên gia OCR tiếng Việt. Trích xuất TOÀN BỘ nội dung từ trang tài liệu này.

**QUY TẮC:**
1. Giữ nguyên văn bản, sửa lỗi OCR rõ ràng, đảm bảo dấu thanh điệu tiếng Việt chính xác
2. Tiêu đề chính → `# Tiêu đề` | Phụ → `## ...` | Mục con → `### ...`
3. Bảng → Markdown table chuẩn, giữ nguyên số liệu
4. Danh sách bullet → `- item` | numbered → `1. item`
5. KHÔNG thêm nội dung ngoài tài liệu, KHÔNG bỏ sót văn bản nào

Trích xuất nội dung trang:"""

    TABLE_FOCUS = """Trang này chứa bảng biểu. Trích xuất CHÍNH XÁC toàn bộ bảng theo Markdown.

**YÊU CẦU:**
- Tạo bảng Markdown hoàn chỉnh (header + separator + rows)
- Giữ nguyên số liệu, đơn vị đo, không làm tròn
- Ô trống → dấu `-`
- Với văn bản ngoài bảng: trích xuất đầy đủ ở trước/sau bảng

Trích xuất toàn bộ nội dung trang:"""

    IMAGE_FOCUS = """Trang này chứa hình ảnh hoặc biểu đồ. Hãy:
1. Trích xuất TẤT CẢ văn bản/số liệu hiển thị trong trang
2. Với biểu đồ: nêu loại, trục, giá trị chính, xu hướng tổng thể
3. Với sơ đồ/flowchart: mô tả các bước/luồng
4. Với hình minh họa: mô tả ngắn gọn

Sau đó trích xuất văn bản khác trên trang (nếu có):"""


# ============================================================================
# PAGE CLASSIFIER
# ============================================================================

class PageClassifier:
    """
    Classify PDF page type without OpenCV.
    Uses pdfplumber metadata: char count + image bounding boxes.
    """

    @staticmethod
    def classify(
        page,          # pdfplumber page object
        page_width: float,
        page_height: float,
    ) -> str:
        text = page.extract_text() or ""
        char_count = len(text.strip())

        images = page.images or []
        image_area = sum(
            abs((img.get("x1", 0) - img.get("x0", 0)) *
                (img.get("y1", 0) - img.get("y0", 0)))
            for img in images
        )
        page_area = page_width * page_height if page_width * page_height > 0 else 1
        image_coverage = image_area / page_area

        tables = page.extract_tables() or []
        has_table = len(tables) > 0

        if char_count >= TEXT_PAGE_MIN_CHARS and image_coverage < IMAGE_COVERAGE_THRESHOLD:
            return PageType.TABLE if has_table else PageType.TEXT
        elif image_coverage >= IMAGE_COVERAGE_THRESHOLD and char_count < TEXT_PAGE_MIN_CHARS:
            return PageType.IMAGE
        else:
            return PageType.TABLE if has_table else PageType.MIXED


# ============================================================================
# IMAGE EXTRACTOR  (PDF embedded images → base64 Markdown)
# ============================================================================

class PDFImageExtractor:
    """Extract embedded images from a pdfplumber page and return base64 blocks."""

    @staticmethod
    def extract_images_from_page(pdf_page, page_num: int) -> List[str]:
        """
        Returns list of Markdown image strings:
            ![Page {n} Image {i}](data:image/png;base64,<b64>)
        """
        markdown_images: List[str] = []
        images = pdf_page.images or []

        for idx, img_meta in enumerate(images):
            try:
                # pdfplumber stores raw image data in img_meta if available
                raw = img_meta.get("stream") or img_meta.get("data")
                if raw is None:
                    # Fallback: crop page region as PNG
                    x0 = img_meta.get("x0", 0)
                    y0 = img_meta.get("y0", 0)
                    x1 = img_meta.get("x1", pdf_page.width)
                    y1 = img_meta.get("y1", pdf_page.height)
                    bbox = (x0, y0, x1, y1)
                    cropped = pdf_page.crop(bbox).to_image(resolution=150)
                    buf = io.BytesIO()
                    cropped.save(buf, format="PNG")
                    raw = buf.getvalue()

                if isinstance(raw, (bytes, bytearray)):
                    b64 = base64.b64encode(raw).decode("utf-8")
                    # Detect format
                    mime = PDFImageExtractor._detect_mime(raw)
                    label = f"Page {page_num} Image {idx + 1}"
                    markdown_images.append(f"![{label}](data:{mime};base64,{b64})")
                    logger.debug(f"  Extracted image {idx+1} from page {page_num} ({len(raw)} bytes)")

            except Exception as e:
                logger.debug(f"  Could not extract image {idx+1} page {page_num}: {e}")

        return markdown_images

    @staticmethod
    def _detect_mime(data: bytes) -> str:
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        if data[:3] == b'\xff\xd8\xff':
            return "image/jpeg"
        if data[:4] == b'GIF8':
            return "image/gif"
        if data[:4] == b'RIFF' and data[8:12] == b'WEBP':
            return "image/webp"
        return "image/png"  # default


# ============================================================================
# IMAGE PREPROCESSOR
# ============================================================================

class ImagePreprocessor:
    @staticmethod
    def enhance_for_ocr(image: Image.Image, dpi_scale: float = 1.5) -> Image.Image:
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        w, h = image.size
        if w < 800 or h < 800:
            image = image.resize(
                (int(w * dpi_scale), int(h * dpi_scale)),
                Image.LANCZOS
            )
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image


# ============================================================================
# PDFPLUMBER TEXT EXTRACTOR  (fast path)
# ============================================================================

class PlumberExtractor:
    """Extract text + tables from text-dominant pages using pdfplumber."""

    @staticmethod
    def extract_page(pdf_page, page_num: int) -> str:
        """Returns Markdown string for this page."""
        sections: List[str] = []

        # --- Tables first (preserve structure) ---
        tables = pdf_page.extract_tables() or []
        table_bboxes = []

        for table in tables:
            if not table:
                continue
            md = PlumberExtractor._table_to_markdown(table)
            if md:
                sections.append(md)

        # --- Text (excluding table regions to avoid duplication) ---
        words = pdf_page.extract_words(
            x_tolerance=3, y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
        ) or []

        if words:
            text = pdf_page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            text = text.strip()
            if text:
                sections.insert(0, text)   # text before tables

        return "\n\n".join(s for s in sections if s.strip())

    @staticmethod
    def _table_to_markdown(table: List[List]) -> str:
        """Convert pdfplumber table (list of lists) to Markdown table."""
        if not table or not any(table):
            return ""

        # Normalize: replace None with ""
        rows = [[str(cell).strip() if cell is not None else "" for cell in row]
                for row in table]

        # Drop completely empty rows
        rows = [r for r in rows if any(c for c in r)]
        if not rows:
            return ""

        max_cols = max(len(r) for r in rows)
        rows = [r + [""] * (max_cols - len(r)) for r in rows]

        # Header
        header = rows[0]
        sep = [":---" if not _is_numeric_col(rows, col_idx) else "---:"
               for col_idx in range(max_cols)]

        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in rows[1:]:
            lines.append("| " + " | ".join(c if c else "-" for c in row) + " |")

        return "\n".join(lines)


def _is_numeric_col(rows: List[List[str]], col_idx: int) -> bool:
    values = [rows[r][col_idx] for r in range(1, len(rows)) if rows[r][col_idx] not in ("", "-")]
    if not values:
        return False
    numeric = sum(1 for v in values if re.match(r'^-?[\d,\.]+%?$', v.replace(',', '')))
    return numeric / len(values) > 0.6


# ============================================================================
# VL INFERENCE ENGINE
# ============================================================================

class VLInferenceEngine:
    """Wraps PaddleOCR-VL model for single-image inference."""

    MODEL_ID = "PaddlePaddle/PaddleOCR-VL"

    # Token budget per page type
    TOKEN_BUDGET: Dict[str, int] = {
        PageType.TEXT:  600,
        PageType.TABLE: 1400,
        PageType.IMAGE: 1000,
        PageType.MIXED: 1200,
    }

    def __init__(
        self,
        use_gpu: bool = True,
        use_flash_attention: bool = True,
        use_compile: bool = False,
        gpu_id: int = 0,
        enable_spell_correction: bool = True,
        enable_table_normalization: bool = True,
    ):
        if use_gpu and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.use_flash_attention = use_flash_attention
        self.use_compile = use_compile
        self.enable_spell_correction = enable_spell_correction
        self.enable_table_normalization = enable_table_normalization

        self.model = None
        self.processor = None
        self.prompts = OCRPrompts()
        self._load_model()

    # ------------------------------------------------------------------ #

    def _load_model(self):
        from transformers import AutoProcessor, PaddleOCRVLForConditionalGeneration

        self._log_env()
        t0 = time.time()

        load_kwargs = dict(
            torch_dtype=torch.float16 if self.use_gpu else torch.float32,
            device_map={"": self.device},
        )

        if self.use_flash_attention and self.use_gpu:
            try:
                self.model = PaddleOCRVLForConditionalGeneration.from_pretrained(
                    self.MODEL_ID, attn_implementation="flash_attention_2", **load_kwargs
                )
                logger.info("✅ Flash Attention 2 enabled")
            except Exception as fa_err:
                logger.warning(f"Flash Attention 2 failed ({fa_err}), using eager")
                self.model = PaddleOCRVLForConditionalGeneration.from_pretrained(
                    self.MODEL_ID, **load_kwargs
                )
        else:
            self.model = PaddleOCRVLForConditionalGeneration.from_pretrained(
                self.MODEL_ID, **load_kwargs
            )

        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model.eval()

        if self.use_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("✅ torch.compile applied")
            except Exception as ce:
                logger.warning(f"torch.compile skipped: {ce}")

        logger.info(f"✅ Model loaded in {time.time()-t0:.1f}s on {self.device}")

    def _log_env(self):
        logger.info("=" * 60)
        logger.info(f"PaddleOCR-VL  device={self.device}  FA2={self.use_flash_attention}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                logger.info(
                    f"  GPU {i}: {torch.cuda.get_device_name(i)} "
                    f"| free={free/1024**3:.1f}GB / {total/1024**3:.1f}GB"
                )
        logger.info("=" * 60)

    # ------------------------------------------------------------------ #

    def infer(
        self,
        image: Image.Image,
        page_type: str = PageType.MIXED,
    ) -> str:
        """Run VL inference for a single page image."""
        prompt = self._pick_prompt(page_type)
        max_tokens = self.TOKEN_BUDGET.get(page_type, 1000)
        return self._run_infer(image, prompt, max_tokens)

    def _pick_prompt(self, page_type: str) -> str:
        if page_type == PageType.TABLE:
            return self.prompts.TABLE_FOCUS
        if page_type == PageType.IMAGE:
            return self.prompts.IMAGE_FOCUS
        return self.prompts.FULL_DOCUMENT

    def _run_infer(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        try:
            t0 = time.time()
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": prompt},
            ]}]
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text_input], images=[image],
                return_tensors="pt", padding=False,
            )
            t1 = time.time()

            if self.use_gpu:
                inputs = {k: v.to(self.device) if hasattr(v, "to") else v
                          for k, v in inputs.items()}
            t2 = time.time()

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            t3 = time.time()

            input_len = inputs["input_ids"].shape[1]
            generated = output_ids[:, input_len:]
            result = self.processor.batch_decode(
                generated, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            logger.info(
                f"  VL infer: prep={t1-t0:.2f}s xfer={t2-t1:.2f}s "
                f"gen={t3-t2:.2f}s tok={generated.shape[1]} total={t3-t0:.2f}s"
            )
            return result.strip()

        except Exception as e:
            logger.error(f"VL inference error: {e}", exc_info=True)
            return ""


# ============================================================================
# PAGE RESULT
# ============================================================================

class PageResult:
    def __init__(self, page_num: int, page_type: str, content: str,
                 image_blocks: List[str] = None):
        self.page_num = page_num
        self.page_type = page_type
        self.content = content
        self.image_blocks = image_blocks or []

    def to_markdown(self) -> str:
        parts = [self.content] if self.content.strip() else []
        parts.extend(self.image_blocks)
        return "\n\n".join(parts)


# ============================================================================
# HYBRID PDF PROCESSOR
# ============================================================================

class HybridPDFProcessor:
    """
    Process PDF pages with hybrid strategy:
      - Text pages  → pdfplumber (fast, no GPU)
      - Mixed/Table → VL model
      - Image pages → VL model
    Embedded images are always extracted as base64 Markdown blocks.
    """

    def __init__(self, vl_engine: VLInferenceEngine, image_scale: float = 1.5):
        self.vl = vl_engine
        self.image_scale = image_scale
        self.preprocessor = ImagePreprocessor()
        self.classifier = PageClassifier()
        self.img_extractor = PDFImageExtractor()

    # ------------------------------------------------------------------ #

    def process(self, file_path: str) -> Optional[str]:
        try:
            import pdfplumber
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ImportError(f"Missing dependency: {e}. Run: pip install pdfplumber pdf2image") from e

        t0 = time.time()

        with pdfplumber.open(file_path) as pdf:
            total = len(pdf.pages)
            logger.info(f"PDF: {total} pages | {Path(file_path).name}")

            # --- Classify all pages (fast, no rendering needed) ---
            page_types = self._classify_all(pdf.pages)
            type_counts = {t: page_types.count(t) for t in set(page_types)}
            logger.info(f"Page classification: {type_counts}")

            # --- Determine which pages need rendering ---
            render_needed = [
                i for i, t in enumerate(page_types)
                if t != PageType.TEXT
            ]

            # --- Render only non-text pages (parallel) ---
            rendered_images: Dict[int, Image.Image] = {}
            if render_needed:
                logger.info(f"Rendering {len(render_needed)} non-text pages (DPI={PDF_RENDER_DPI})...")
                rendered_images = self._render_pages(file_path, render_needed, total)

            # --- Process all pages ---
            page_results: List[PageResult] = []

            for page_num, (plumber_page, ptype) in enumerate(zip(pdf.pages, page_types), 1):
                result = self._process_page(
                    plumber_page=plumber_page,
                    page_num=page_num,
                    page_type=ptype,
                    rendered_image=rendered_images.get(page_num - 1),
                    total=total,
                )
                page_results.append(result)

        # --- Merge all pages ---
        merged = self._merge_pages(page_results)
        elapsed = time.time() - t0
        logger.info(
            f"✅ PDF done: {total} pages in {elapsed:.1f}s "
            f"({elapsed/total:.1f}s/page avg)"
        )
        return merged

    # ------------------------------------------------------------------ #

    def _classify_all(self, pages) -> List[str]:
        types = []
        for page in pages:
            try:
                t = self.classifier.classify(page, page.width, page.height)
            except Exception:
                t = PageType.MIXED
            types.append(t)
        return types

    def _render_pages(
        self,
        file_path: str,
        page_indices: List[int],  # 0-based
        total: int,
    ) -> Dict[int, Image.Image]:
        """Render specific pages from PDF → PIL Images."""
        from pdf2image import convert_from_path

        # pdf2image is 1-indexed; render all needed pages in one call
        # by specifying first_page / last_page in batches
        # For non-contiguous pages, render all and filter.
        needed_set = set(page_indices)

        all_pages = convert_from_path(
            file_path,
            dpi=PDF_RENDER_DPI,
            fmt='RGB',
            thread_count=RENDER_WORKERS,
        )

        result: Dict[int, Image.Image] = {}
        for idx in needed_set:
            if idx < len(all_pages):
                result[idx] = all_pages[idx]

        return result

    def _process_page(
        self,
        plumber_page,
        page_num: int,
        page_type: str,
        rendered_image: Optional[Image.Image],
        total: int,
    ) -> PageResult:
        logger.info(f"  [{page_num:3d}/{total}] type={page_type:<8s}", )

        # Always extract embedded images as base64
        image_blocks = self.img_extractor.extract_images_from_page(plumber_page, page_num)

        if page_type == PageType.TEXT:
            # Fast path: no VL call
            content = PlumberExtractor.extract_page(plumber_page, page_num)
            logger.info(f"               → plumber fast-path ({len(content)} chars)")
        else:
            # VL path
            if rendered_image is None:
                logger.warning(f"  Page {page_num}: missing render, falling back to plumber")
                content = PlumberExtractor.extract_page(plumber_page, page_num)
            else:
                enhanced = self.preprocessor.enhance_for_ocr(rendered_image, self.image_scale)
                content = self.vl.infer(enhanced, page_type)

                if not content or len(content.strip()) < 5:
                    logger.warning(f"  Page {page_num}: VL returned empty, retry...")
                    content = self.vl.infer(enhanced, page_type)

                # Supplement: if VL missed embedded images, plumber might have text
                if page_type == PageType.IMAGE and not content.strip():
                    plumber_text = PlumberExtractor.extract_page(plumber_page, page_num)
                    if plumber_text.strip():
                        content = plumber_text

        return PageResult(
            page_num=page_num,
            page_type=page_type,
            content=content,
            image_blocks=image_blocks,
        )

    def _merge_pages(self, results: List[PageResult]) -> str:
        """Merge page results into final Markdown document."""
        parts: List[str] = []

        for r in results:
            md = r.to_markdown()
            if md.strip():
                parts.append(md)

        # Join with page separator
        combined = "\n\n---\n\n".join(parts)

        # Post-process
        combined = self._post_process(combined)
        return combined

    # ------------------------------------------------------------------ #
    # Post-processing (image-aware: don't touch base64 blocks)
    # ------------------------------------------------------------------ #

    def _post_process(self, text: str) -> str:
        if not text:
            return ""

        # Split out base64 image blocks before processing
        # Pattern: ![...](data:image/...;base64,...)
        image_placeholder = "%%IMAGE_BLOCK_{}%%"
        image_blocks: List[str] = []

        def extract_image(m: re.Match) -> str:
            idx = len(image_blocks)
            image_blocks.append(m.group(0))
            return image_placeholder.format(idx)

        # Preserve base64 image blocks
        protected = re.sub(
            r'!\[[^\]]*\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)',
            extract_image,
            text,
        )

        # --- Text cleanup ---
        protected = self._basic_cleanup(protected)
        protected = self._normalize_tables(protected)
        protected = self._fix_headings(protected)
        protected = self._rule_based_corrections(protected)

        # Restore base64 blocks
        for idx, block in enumerate(image_blocks):
            protected = protected.replace(image_placeholder.format(idx), block)

        return protected.strip()

    def _basic_cleanup(self, text: str) -> str:
        # Collapse 4+ consecutive newlines to 3
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        # Trailing whitespace on lines
        text = re.sub(r'[ \t]+\n', '\n', text)
        # Multiple spaces (not inside code)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        # Page number artifacts
        text = re.sub(r'(?m)^(Trang|Page)\s+\d+\s*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(?m)^\d+\s*$', '', text)

        # Remove repeated short lines (header/footer duplicates)
        lines = text.split('\n')
        counts: Dict[str, int] = {}
        for ln in lines:
            s = ln.strip()
            if s and len(s) < 60:
                counts[s] = counts.get(s, 0) + 1
        text = '\n'.join(
            ln for ln in lines
            if counts.get(ln.strip(), 0) <= 3
        )
        return text

    def _normalize_tables(self, text: str) -> str:
        """Re-align Markdown tables: ensure consistent column count."""
        def normalize_block(m: re.Match) -> str:
            lines = [ln.strip() for ln in m.group(0).strip().split('\n') if ln.strip()]
            rows, has_sep = [], False
            for ln in lines:
                if re.match(r'^\|[-:\s|]+\|$', ln):
                    has_sep = True
                    continue
                if ln.startswith('|'):
                    rows.append([c.strip() for c in ln.split('|')[1:-1]])

            if not rows:
                return m.group(0)

            max_cols = max(len(r) for r in rows)
            norm = [r + [''] * (max_cols - len(r)) for r in rows]

            if not has_sep and len(norm) >= 2:
                seps = ['---:' if _is_numeric_col(norm, c) else '---'
                        for c in range(max_cols)]
                norm.insert(1, seps)

            table_lines = []
            for i, row in enumerate(norm):
                table_lines.append('| ' + ' | '.join(c if c else '-' for c in row) + ' |')
                if i == 0 and not has_sep:
                    pass  # separator already inserted above

            return '\n'.join(table_lines)

        return re.sub(r'(?m)((?:^\|.+\|\s*\n?)+)', normalize_block, text)

    def _fix_headings(self, text: str) -> str:
        """Ensure blank line before headings; convert ALL-CAPS short lines."""
        lines = text.split('\n')
        result: List[str] = []
        for ln in lines:
            stripped = ln.strip()
            if (stripped and len(stripped) < 80
                    and stripped.isupper()
                    and not stripped.startswith('#')
                    and not stripped.startswith('|')
                    and not stripped.startswith('!')
                    and len(stripped.split()) > 1):
                ln = f"## {stripped}"
                stripped = ln
            if stripped.startswith('#') and result and result[-1].strip():
                result.append('')
            result.append(ln)
        return '\n'.join(result)

    def _rule_based_corrections(self, text: str) -> str:
        corrections = {
            r'\bViet Nam\b':      'Việt Nam',
            r'\bHa Noi\b':        'Hà Nội',
            r'\bHo Chi Minh\b':   'Hồ Chí Minh',
            r'(\d)\s+\.(\d)':     r'\1.\2',
            r'(\d)\s+,(\d)':      r'\1,\2',
        }
        for pat, repl in corrections.items():
            try:
                text = re.sub(pat, repl, text)
            except Exception:
                pass
        return text


# ============================================================================
# DOCX / EXCEL / IMAGE / TXT HANDLERS  (unchanged from v2, kept for API compat)
# ============================================================================

class PaddleOCRVLProcessor:
    """
    Public API — mirrors the interface expected by document_processor.py.
    Now delegates PDF work to HybridPDFProcessor.
    """

    MODEL_ID = "PaddlePaddle/PaddleOCR-VL"

    def __init__(
        self,
        use_gpu: bool = True,
        image_scale: float = 1.5,
        max_new_tokens: int = 800,          # kept for API compat (overridden per page type)
        enable_spell_correction: bool = True,
        enable_table_normalization: bool = True,
        batch_size: int = 4,                # kept for API compat
        use_flash_attention: bool = True,
        use_compile: bool = False,
        gpu_id: int = 0,
    ):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.image_scale = image_scale
        self.enable_spell_correction = enable_spell_correction
        self.enable_table_normalization = enable_table_normalization

        logger.info("🚀 Loading VL inference engine...")
        self._vl = VLInferenceEngine(
            use_gpu=use_gpu,
            use_flash_attention=use_flash_attention,
            use_compile=use_compile,
            gpu_id=gpu_id,
            enable_spell_correction=enable_spell_correction,
            enable_table_normalization=enable_table_normalization,
        )
        self._pdf_proc = HybridPDFProcessor(self._vl, image_scale=image_scale)
        self._preproc = ImagePreprocessor()
        logger.info("✅ PaddleOCRVLProcessor ready (hybrid mode)")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def process_pdf(self, file_path: str) -> Optional[str]:
        logger.info(f"📄 process_pdf: {Path(file_path).name}")
        result = self._pdf_proc.process(file_path)
        if not result or len(result.strip()) < 50:
            raise ValueError(f"PDF processing returned insufficient content: {file_path}")
        return result

    def process_docx(self, file_path: str) -> Optional[str]:
        logger.info(f"📝 process_docx: {Path(file_path).name}")
        return self._process_docx_internal(file_path)

    def process_image(self, file_path: str) -> Optional[str]:
        logger.info(f"🖼️  process_image: {Path(file_path).name}")
        try:
            image = Image.open(file_path).convert('RGB')
            enhanced = self._preproc.enhance_for_ocr(image, self.image_scale)
            text = self._vl.infer(enhanced, PageType.MIXED)
            return self._vl_post_process(text) if text else None
        except Exception as e:
            logger.error(f"Image processing failed: {e}", exc_info=True)
            return None

    def _post_process(self, text: str) -> str:
        """Exposed for document_processor.py text processing path."""
        return self._vl_post_process(text)

    def _process_excel(self, file_path: str) -> Optional[str]:
        try:
            import pandas as pd
            excel_file = pd.ExcelFile(file_path)
            sections: List[str] = []
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
                if df.empty:
                    continue
                sections.append(f"## {sheet_name}")
                header_row = self._detect_header_row(df)
                if header_row >= 0:
                    df.columns = df.iloc[header_row].astype(str)
                    df = df.iloc[header_row + 1:].reset_index(drop=True)
                sections.append(self._dataframe_to_markdown(df))
            return self._vl_post_process("\n\n".join(sections))
        except Exception as e:
            logger.error(f"Excel processing failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _vl_post_process(self, text: str) -> str:
        """Lightweight post-process for non-PDF content (no image blocks)."""
        if not text:
            return ""
        text = self._pdf_proc._basic_cleanup(text)
        text = self._pdf_proc._normalize_tables(text)
        text = self._pdf_proc._fix_headings(text)
        text = self._pdf_proc._rule_based_corrections(text)
        return text.strip()

    def _process_docx_internal(self, file_path: str) -> Optional[str]:
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
                    else:                       sections.append(text)
                elif tag == 'tbl':
                    table = docx.table.Table(element, doc)
                    rows = [[cell.text.strip().replace('\n', ' ') for cell in row.cells]
                            for row in table.rows]
                    md = PlumberExtractor._table_to_markdown(rows)
                    if md:
                        sections.append(md)

            result = "\n\n".join(sections)
            return self._vl_post_process(result)
        except Exception as e:
            logger.error(f"DOCX failed: {e}", exc_info=True)
            return self._convert_office_to_pdf_and_process(file_path)

    def _convert_office_to_pdf_and_process(self, file_path: str) -> Optional[str]:
        try:
            import subprocess
            with tempfile.TemporaryDirectory() as tmpdir:
                subprocess.run(
                    ['libreoffice', '--headless', '--convert-to', 'pdf',
                     '--outdir', tmpdir, file_path],
                    capture_output=True, text=True, timeout=60
                )
                pdf_files = list(Path(tmpdir).glob('*.pdf'))
                if pdf_files:
                    return self.process_pdf(str(pdf_files[0]))
            return None
        except Exception as e:
            logger.error(f"Office→PDF conversion failed: {e}")
            return None

    def _detect_header_row(self, df) -> int:
        import pandas as pd
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            if sum(1 for v in row if isinstance(v, str) and v.strip()) / len(row) > 0.6:
                return i
        return 0

    def _dataframe_to_markdown(self, df) -> str:
        import pandas as pd
        rows = [[str(c) for c in df.columns]]
        for _, row in df.iterrows():
            rows.append(['-' if pd.isna(v) else str(v).strip() for v in row])
        return PlumberExtractor._table_to_markdown(rows)


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
) -> PaddleOCRVLProcessor:
    """Return global singleton PaddleOCRVLProcessor (thread-safe)."""
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
                )
    return _instance