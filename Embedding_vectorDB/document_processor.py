#!/usr/bin/env python3
"""
document_processor.py — Vietnamese Document Pipeline (v4)
==========================================================
Nâng cấp so với v3:

ENGINE THAY ĐỔI:
  - PaddleOCR v3.7.0 + PP-StructureV3 thay vì PaddleOCR-VL HuggingFace model
  - PP-OCRv5 recognition model (tốt hơn, nhẹ hơn)
  - ProtonX Vietnamese text correction tích hợp tại nguồn (trong paddle_ocr_processor)

TABLE CHUNKING (giữ nguyên v3, cải tiến thêm):
  - PP-StructureV3 native table detection → HTML → Markdown (độ chính xác cao hơn)
  - Detect thêm TSV và aligned-text tables
  - Bảng luôn là 1 chunk nguyên vẹn

TOC/ARTIFACT/HEADING (giữ nguyên v3)
  - TOC chunks gộp thành 1 chunk type="toc"
  - OCR artifact lines lọc sạch
  - Heading dedup

OVERSIZED CHUNK FIX (giữ nguyên v3)
  - Hard cap không để chunk > max_chunk_size

CONFIG:
  - Thêm correction_model: str = "student" (teacher/student/nano)
  - Thêm lang: str = "vi"
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ChunkRecord:
    """One chunk ready to insert into document_chunks collection."""

    # Identity
    document_id: str = ""
    chunk_index: int = 0
    chunk_mode: str = "smart"

    # Content
    content: str = ""
    content_with_ctx: str = ""

    # Context / Structure
    section_path: str = ""
    context_header: str = ""
    chunk_type: str = "complete_section"
    page_num: int = 0
    part_index: int = 0
    total_parts: int = 1

    # Metrics
    token_count: int = 0
    char_count: int = 0

    # Flags (INT8 in Milvus)
    has_table: int = 0
    has_image: int = 0
    has_heading: int = 0
    is_overlap: int = 0

    # Vector (filled by EmbeddingService)
    content_vector: Optional[List[float]] = field(default=None, repr=False)

    @property
    def milvus_id(self) -> str:
        safe_doc = self.document_id[:80]
        return f"{safe_doc}__{self.chunk_mode}__{self.chunk_index:04d}"

    def to_milvus_dict(self) -> Dict:
        return {
            "id":               self.milvus_id[:220],
            "document_id":      self.document_id[:100],
            "content":          self.content[:8000],
            "content_with_ctx": self.content_with_ctx[:10000],
            "section_path":     self.section_path[:500],
            "context_header":   self.context_header[:1000],
            "chunk_type":       self.chunk_type[:30],
            "page_num":         int(self.page_num),
            "part_index":       int(self.part_index),
            "total_parts":      int(self.total_parts),
            "token_count":      int(self.token_count),
            "char_count":       int(self.char_count),
            "has_table":        int(self.has_table),
            "has_image":        int(self.has_image),
            "has_heading":      int(self.has_heading),
            "is_overlap":       int(self.is_overlap),
            "content_vector":   self.content_vector or [],
        }


# ============================================================================
# REGEX PATTERNS
# ============================================================================

_BASE64_IMG_RE = re.compile(
    r'!\[[^\]]*\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)',
    re.DOTALL,
)
_MARKDOWN_TABLE_RE = re.compile(r'(?m)((?:^\|.+\|\s*\n?){2,})')
_TSV_TABLE_RE = re.compile(
    r'(?m)((?:^[^\n\t|]+(?:\t[^\n\t]+){2,}\s*\n){2,})',
)
_NOISE_LINE_RE = re.compile(r'^[\s\*\-_=\.#~]{5,}$')
_TOC_ENTRY_RE = re.compile(r'\.{3,}\s*\d+\s*$')
_PAGE_SEP_RE = re.compile(r'\n\n---\n\n|\f')
_HEADING_RE = re.compile(r'^#{1,6}\s+(.+)$')
_SENTENCE_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠ-ỸẮẰẶẤẦẨẪẬẮẰẴẶ\d])'
)
_WORD_RE = re.compile(r'\w+|[^\w\s]')


# ============================================================================
# HEADING DETECTOR
# ============================================================================

def _is_heading_line(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 120:
        return False
    if _HEADING_RE.match(s):
        return True
    if s.endswith(':') and len(s) < 60 and len(s.split()) <= 8:
        return True
    clean = re.sub(r'[0-9\s\-_./,:;()]', '', s)
    if clean and clean == clean.upper() and len(s.split()) >= 2 and len(s) < 80:
        return True
    return False


def _line_to_heading_text(line: str) -> str:
    s = line.strip()
    m = _HEADING_RE.match(s)
    if m:
        return m.group(1).strip()
    return s.rstrip(':').strip()


# ============================================================================
# TOKEN ESTIMATOR
# ============================================================================

def estimate_tokens(text: str) -> int:
    clean = _BASE64_IMG_RE.sub('[IMAGE]', text)
    return len(_WORD_RE.findall(clean))


# ============================================================================
# PAGE SPLITTER
# ============================================================================

def split_by_pages(markdown: str) -> List[Tuple[int, str]]:
    parts = _PAGE_SEP_RE.split(markdown)
    return [(i + 1, p) for i, p in enumerate(parts) if p.strip()]


# ============================================================================
# TABLE NORMALIZER
# ============================================================================

class TableNormalizer:
    """Detect và normalize các dạng bảng phi-Markdown về Markdown chuẩn."""

    @staticmethod
    def is_markdown_table(text: str) -> bool:
        return bool(_MARKDOWN_TABLE_RE.search(text))

    @staticmethod
    def is_tsv_block(lines: List[str]) -> bool:
        if len(lines) < 2:
            return False
        tab_lines = sum(1 for l in lines if l.count('\t') >= 2)
        return tab_lines >= 2 and tab_lines / len(lines) >= 0.7

    @staticmethod
    def is_aligned_table(lines: List[str]) -> bool:
        if len(lines) < 2:
            return False
        _MULTI_SPACE_RE = re.compile(r'\s{2,}')
        aligned = sum(
            1 for l in lines if len(_MULTI_SPACE_RE.findall(l.strip())) >= 2
        )
        return aligned >= 2 and aligned / len(lines) >= 0.6

    @staticmethod
    def tsv_to_markdown(text: str) -> str:
        lines = [l for l in text.strip().split('\n') if l.strip()]
        if not lines:
            return text

        rows = [l.split('\t') for l in lines]
        max_cols = max(len(r) for r in rows)
        rows = [r + [''] * (max_cols - len(r)) for r in rows]
        rows = [[c.strip() for c in row] for row in rows]

        def _is_numeric(col_idx: int) -> bool:
            vals = [rows[r][col_idx] for r in range(1, len(rows))
                    if rows[r][col_idx] not in ('', '-')]
            if not vals:
                return False
            num = sum(1 for v in vals
                      if re.match(r'^-?[\d,\.]+%?$', v.replace(',', '')))
            return num / len(vals) > 0.6

        header = rows[0]
        sep = ['---:' if _is_numeric(i) else ':---' for i in range(max_cols)]
        md_lines = [
            '| ' + ' | '.join(header) + ' |',
            '| ' + ' | '.join(sep) + ' |',
        ]
        for row in rows[1:]:
            md_lines.append('| ' + ' | '.join(c if c else '-' for c in row) + ' |')
        return '\n'.join(md_lines)

    @staticmethod
    def aligned_to_markdown(text: str) -> str:
        lines = [l.rstrip() for l in text.strip().split('\n') if l.strip()]
        if len(lines) < 2:
            return text

        max_len = max(len(l) for l in lines)
        padded = [l.ljust(max_len) for l in lines]

        gap_cols: List[int] = [
            col for col in range(max_len)
            if all(l[col] == ' ' for l in padded)
        ]

        if not gap_cols:
            return text

        splits: List[int] = [0]
        in_gap = False
        gap_set = set(gap_cols)
        for col in range(max_len):
            if col in gap_set:
                if not in_gap:
                    splits.append(col)
                    in_gap = True
            else:
                in_gap = False
        splits.append(max_len)

        rows: List[List[str]] = []
        for line in lines:
            cells = []
            for i in range(len(splits) - 1):
                cell = line[splits[i]:splits[i+1]].strip()
                if cell:
                    cells.append(cell)
            if cells:
                rows.append(cells)

        if not rows or max(len(r) for r in rows) < 2:
            return text

        max_cols = max(len(r) for r in rows)
        rows = [r + [''] * (max_cols - len(r)) for r in rows]

        md_lines = [
            '| ' + ' | '.join(rows[0]) + ' |',
            '| ' + ' | '.join(['---'] * max_cols) + ' |',
        ]
        for row in rows[1:]:
            md_lines.append('| ' + ' | '.join(c if c else '-' for c in row) + ' |')
        return '\n'.join(md_lines)

    @classmethod
    def detect_and_normalize(cls, text: str) -> Tuple[bool, str]:
        stripped = text.strip()
        if cls.is_markdown_table(stripped):
            return True, stripped

        lines = [l for l in stripped.split('\n') if l.strip()]

        if cls.is_tsv_block(lines):
            return True, cls.tsv_to_markdown(stripped)

        if cls.is_aligned_table(lines):
            normalized = cls.aligned_to_markdown(stripped)
            if '|' in normalized:
                return True, normalized

        return False, text


# ============================================================================
# TOC DETECTOR
# ============================================================================

def _is_toc_chunk(text: str) -> bool:
    lines = [l for l in text.split('\n') if l.strip()]
    if len(lines) < 3:
        return False
    toc_count = sum(1 for l in lines if _TOC_ENTRY_RE.search(l))
    return toc_count / len(lines) >= 0.40


# ============================================================================
# NOISE LINE FILTER
# ============================================================================

def _clean_ocr_artifacts(text: str) -> str:
    cleaned_lines = []
    for line in text.split('\n'):
        if _NOISE_LINE_RE.match(line.strip()):
            if cleaned_lines and cleaned_lines[-1] != '':
                cleaned_lines.append('')
        else:
            cleaned_lines.append(line)

    result_lines: List[str] = []
    blank_count = 0
    for line in cleaned_lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                result_lines.append(line)
        else:
            blank_count = 0
            result_lines.append(line)

    return '\n'.join(result_lines)


# ============================================================================
# SMART CHUNKER v4
# ============================================================================

class SmartChunker:
    """
    Chunker v4 — giữ nguyên logic v3, tương thích với PP-StructureV3 output.
    """

    def __init__(
        self,
        target_chunk_size: int = 450,
        min_chunk_size: int = 80,
        max_chunk_size: int = 700,
        overlap_size: int = 80,
        chunk_mode: str = "smart",
    ):
        self.target = target_chunk_size
        self.min_size = min_chunk_size
        self.max_size = max_chunk_size
        self.overlap_size = overlap_size
        self.chunk_mode = chunk_mode
        self._normalizer = TableNormalizer()

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    def chunk_document(self, markdown: str, document_id: str) -> List[ChunkRecord]:
        markdown = _clean_ocr_artifacts(markdown)
        pages = split_by_pages(markdown)
        all_chunks: List[ChunkRecord] = []
        heading_stack: List[str] = []

        for page_num, page_text in pages:
            page_chunks = self._process_page(page_text, page_num, heading_stack)
            all_chunks.extend(page_chunks)

        all_chunks = self._merge_toc_chunks(all_chunks)
        self._assign_indices(all_chunks)
        self._compute_total_parts(all_chunks)
        all_chunks = self._add_overlap(all_chunks)

        for c in all_chunks:
            c.document_id = document_id
            c.chunk_mode = self.chunk_mode

        logger.info(
            f"✅ Chunked '{document_id}': {len(all_chunks)} chunks "
            f"from {len(pages)} page(s) | mode={self.chunk_mode}"
        )
        return all_chunks

    # ------------------------------------------------------------------ #
    # PAGE PROCESSING
    # ------------------------------------------------------------------ #

    def _process_page(
        self,
        page_text: str,
        page_num: int,
        heading_stack: List[str],
    ) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        protected_blocks: Dict[str, str] = {}
        text = page_text

        def _protect(pattern: re.Pattern, block_type: str, t: str) -> str:
            def _replace(m: re.Match) -> str:
                key = f"%%{block_type}_{len(protected_blocks):04d}%%"
                protected_blocks[key] = m.group(0)
                return f"\n\n{key}\n\n"
            return pattern.sub(_replace, t)

        text = _protect(_BASE64_IMG_RE, "IMG", text)
        text = _protect(_MARKDOWN_TABLE_RE, "TABLE", text)
        text = _protect(_TSV_TABLE_RE, "TSV", text)

        lines = text.split('\n')
        pending_lines: List[str] = []
        last_heading: str = ""

        def _flush_pending():
            if not pending_lines:
                return
            raw = '\n'.join(pending_lines).strip()
            pending_lines.clear()
            if not raw:
                return
            is_tbl, normalized = self._normalizer.detect_and_normalize(raw)
            if is_tbl and '|' in normalized:
                chunks.append(self._make_table_chunk(
                    normalized, page_num, list(heading_stack)
                ))
                return
            text_chunks = self._chunk_text_block(
                raw, page_num,
                list(heading_stack),
                self._build_context_header(heading_stack),
            )
            chunks.extend(text_chunks)

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("%%IMG_") and stripped.endswith("%%"):
                _flush_pending()
                original = protected_blocks.get(stripped, "")
                if original:
                    chunks.append(self._make_image_chunk(
                        original, page_num, list(heading_stack)
                    ))
                continue

            if stripped.startswith("%%TABLE_") and stripped.endswith("%%"):
                _flush_pending()
                original = protected_blocks.get(stripped, "")
                if original:
                    chunks.append(self._make_table_chunk(
                        original, page_num, list(heading_stack)
                    ))
                continue

            if stripped.startswith("%%TSV_") and stripped.endswith("%%"):
                _flush_pending()
                original = protected_blocks.get(stripped, "")
                if original:
                    normalized = TableNormalizer.tsv_to_markdown(original)
                    chunks.append(self._make_table_chunk(
                        normalized, page_num, list(heading_stack)
                    ))
                continue

            if _is_heading_line(stripped):
                _flush_pending()
                heading_text = _line_to_heading_text(stripped)
                if heading_text and heading_text != last_heading:
                    last_heading = heading_text
                    self._update_heading_stack(heading_stack, stripped, heading_text)
                continue

            if stripped:
                pending_lines.append(stripped)

        _flush_pending()
        return chunks

    # ------------------------------------------------------------------ #
    # HEADING STACK
    # ------------------------------------------------------------------ #

    def _update_heading_stack(
        self,
        stack: List[str],
        raw_line: str,
        heading_text: str,
    ) -> None:
        m = _HEADING_RE.match(raw_line.strip())
        if m:
            level = raw_line.strip().count('#', 0, 7)
            del stack[level - 1:]
            stack.append(heading_text)
        else:
            words = heading_text.split()
            if len(words) <= 3 and stack:
                stack.append(heading_text)
            else:
                stack.clear()
                stack.append(heading_text)

        if len(stack) > 5:
            del stack[:-5]

    def _build_context_header(self, stack: List[str]) -> str:
        if not stack:
            return ""
        return '\n'.join(f"{'#' * (i + 1)} {h}" for i, h in enumerate(stack))

    def _build_section_path(self, stack: List[str]) -> str:
        return ' > '.join(stack) if stack else "Document Root"

    # ------------------------------------------------------------------ #
    # SPECIAL CHUNK FACTORIES
    # ------------------------------------------------------------------ #

    def _make_image_chunk(
        self, image_block: str, page_num: int, heading_stack: List[str]
    ) -> ChunkRecord:
        ctx = self._build_context_header(heading_stack)
        content_with_ctx = f"{ctx}\n\n{image_block}".strip() if ctx else image_block
        return ChunkRecord(
            content=image_block,
            content_with_ctx=content_with_ctx,
            section_path=self._build_section_path(heading_stack),
            context_header=ctx,
            chunk_type="image",
            page_num=page_num,
            token_count=10,
            char_count=len(image_block),
            has_image=1,
            has_heading=int(bool(ctx)),
        )

    def _make_table_chunk(
        self, table_block: str, page_num: int, heading_stack: List[str]
    ) -> ChunkRecord:
        ctx = self._build_context_header(heading_stack)
        content_with_ctx = f"{ctx}\n\n{table_block}".strip() if ctx else table_block
        tokens = estimate_tokens(table_block)
        return ChunkRecord(
            content=table_block,
            content_with_ctx=content_with_ctx,
            section_path=self._build_section_path(heading_stack),
            context_header=ctx,
            chunk_type="table",
            page_num=page_num,
            token_count=tokens,
            char_count=len(table_block),
            has_table=1,
            has_heading=int(bool(ctx)),
        )

    # ------------------------------------------------------------------ #
    # TEXT BLOCK CHUNKING
    # ------------------------------------------------------------------ #

    def _chunk_text_block(
        self,
        text: str,
        page_num: int,
        heading_stack: List[str],
        ctx_header: str,
    ) -> List[ChunkRecord]:
        total_tokens = estimate_tokens(text)
        section_path = self._build_section_path(heading_stack)
        has_table = int(bool(_MARKDOWN_TABLE_RE.search(text)))
        has_image = int(bool(_BASE64_IMG_RE.search(text)))
        has_heading = int(bool(ctx_header))

        def _make(content: str, part_idx: int, ctype: str) -> ChunkRecord:
            c_with_ctx = f"{ctx_header}\n\n{content}".strip() if ctx_header else content
            tokens = estimate_tokens(content)
            return ChunkRecord(
                content=content,
                content_with_ctx=c_with_ctx,
                section_path=section_path,
                context_header=ctx_header,
                chunk_type=ctype,
                page_num=page_num,
                part_index=part_idx,
                token_count=tokens,
                char_count=len(content),
                has_table=has_table,
                has_image=has_image,
                has_heading=has_heading,
            )

        if total_tokens <= self.max_size:
            return [_make(text, 0, "complete_section")]

        return self._split_paragraphs(text, page_num, _make)

    def _split_paragraphs(
        self, text: str, page_num: int, make_fn
    ) -> List[ChunkRecord]:
        paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
        chunks: List[ChunkRecord] = []
        current_parts: List[str] = []
        current_tokens = 0
        part_idx = 0

        def _flush():
            nonlocal part_idx, current_tokens
            if not current_parts:
                return
            joined = '\n\n'.join(current_parts)
            chunks.append(make_fn(joined, part_idx, "partial_section"))
            part_idx += 1
            current_parts.clear()
            current_tokens = 0

        for para in paragraphs:
            p_tokens = estimate_tokens(para)

            if p_tokens > self.max_size:
                _flush()
                sent_chunks = self._split_sentences(para, page_num, make_fn, part_idx)
                chunks.extend(sent_chunks)
                part_idx += len(sent_chunks)
                continue

            if current_tokens + p_tokens > self.target:
                if current_tokens >= self.min_size:
                    _flush()
                current_parts.append(para)
                current_tokens += p_tokens
            else:
                current_parts.append(para)
                current_tokens += p_tokens

        _flush()
        return chunks

    def _split_sentences(
        self, paragraph: str, page_num: int, make_fn, start_part_idx: int
    ) -> List[ChunkRecord]:
        sentences = _SENTENCE_SPLIT_RE.split(paragraph)
        chunks: List[ChunkRecord] = []
        current_parts: List[str] = []
        current_tokens = 0
        part_idx = start_part_idx

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            s_tokens = estimate_tokens(sent)

            if s_tokens > self.max_size:
                if current_parts:
                    chunks.append(make_fn(' '.join(current_parts), part_idx, "sentence_group"))
                    part_idx += 1
                    current_parts = []
                    current_tokens = 0
                char_chunks = self._split_by_char_limit(sent, part_idx, make_fn)
                chunks.extend(char_chunks)
                part_idx += len(char_chunks)
                continue

            if current_tokens + s_tokens > self.target and current_parts:
                chunks.append(make_fn(' '.join(current_parts), part_idx, "sentence_group"))
                part_idx += 1
                current_parts = [sent]
                current_tokens = s_tokens
            else:
                current_parts.append(sent)
                current_tokens += s_tokens

        if current_parts:
            chunks.append(make_fn(' '.join(current_parts), part_idx, "sentence_group"))

        return chunks

    def _split_by_char_limit(
        self, text: str, start_part_idx: int, make_fn
    ) -> List[ChunkRecord]:
        char_limit = self.max_size * 6
        chunks: List[ChunkRecord] = []
        part_idx = start_part_idx

        while text:
            if len(text) <= char_limit:
                chunks.append(make_fn(text, part_idx, "sentence_group"))
                break
            split_at = text.rfind(' ', 0, char_limit)
            if split_at == -1:
                split_at = char_limit
            chunk_text = text[:split_at].strip()
            if chunk_text:
                chunks.append(make_fn(chunk_text, part_idx, "sentence_group"))
                part_idx += 1
            text = text[split_at:].strip()

        return chunks

    # ------------------------------------------------------------------ #
    # TOC CHUNK MERGER
    # ------------------------------------------------------------------ #

    def _merge_toc_chunks(self, chunks: List[ChunkRecord]) -> List[ChunkRecord]:
        result: List[ChunkRecord] = []
        toc_buffer: List[ChunkRecord] = []

        def _flush_toc():
            if not toc_buffer:
                return
            if len(toc_buffer) == 1:
                c = toc_buffer[0]
                c.chunk_type = "toc"
                result.append(c)
            else:
                combined_content = '\n\n'.join(c.content for c in toc_buffer)
                first = toc_buffer[0]
                merged = ChunkRecord(
                    content=combined_content[:8000],
                    content_with_ctx=(
                        f"{first.context_header}\n\n{combined_content}".strip()
                        if first.context_header else combined_content
                    )[:10000],
                    section_path=first.section_path,
                    context_header=first.context_header,
                    chunk_type="toc",
                    page_num=first.page_num,
                    part_index=0,
                    total_parts=1,
                    token_count=estimate_tokens(combined_content),
                    char_count=len(combined_content),
                    has_heading=first.has_heading,
                )
                result.append(merged)
            toc_buffer.clear()

        for chunk in chunks:
            if _is_toc_chunk(chunk.content):
                toc_buffer.append(chunk)
            else:
                _flush_toc()
                result.append(chunk)

        _flush_toc()

        logger.info(
            f"  TOC merge: {len(chunks)} → {len(result)} chunks "
            f"({len(chunks) - len(result)} TOC chunks merged)"
        )
        return result

    # ------------------------------------------------------------------ #
    # POST-PROCESSING
    # ------------------------------------------------------------------ #

    def _assign_indices(self, chunks: List[ChunkRecord]) -> None:
        for i, c in enumerate(chunks):
            c.chunk_index = i

    def _compute_total_parts(self, chunks: List[ChunkRecord]) -> None:
        from collections import defaultdict
        groups: Dict[Tuple, List[ChunkRecord]] = defaultdict(list)
        for c in chunks:
            if c.chunk_type in ("partial_section", "sentence_group"):
                key = (c.section_path, c.page_num, c.chunk_type)
                groups[key].append(c)
        for group in groups.values():
            total = len(group)
            for c in group:
                c.total_parts = total

    def _add_overlap(self, chunks: List[ChunkRecord]) -> List[ChunkRecord]:
        TEXT_TYPES = {"complete_section", "partial_section", "sentence_group"}
        if len(chunks) <= 1:
            return chunks

        result: List[ChunkRecord] = []
        for i, chunk in enumerate(chunks):
            if chunk.chunk_type not in TEXT_TYPES:
                result.append(chunk)
                continue

            if i > 0:
                prev = chunks[i - 1]
                if (prev.chunk_type in TEXT_TYPES
                        and prev.section_path == chunk.section_path
                        and prev.page_num == chunk.page_num):
                    overlap_text = self._tail_tokens(prev.content, self.overlap_size)
                    if overlap_text and overlap_text != chunk.content[:len(overlap_text)]:
                        new_content = f"...{overlap_text}\n\n{chunk.content}"
                        new_ctx = (
                            f"{chunk.context_header}\n\n{new_content}".strip()
                            if chunk.context_header else new_content
                        )
                        chunk = ChunkRecord(**{
                            **chunk.__dict__,
                            "content": new_content[:8000],
                            "content_with_ctx": new_ctx[:10000],
                            "token_count": estimate_tokens(new_content),
                            "char_count": len(new_content),
                            "is_overlap": 1,
                        })
            result.append(chunk)
        return result

    def _tail_tokens(self, text: str, num_tokens: int) -> str:
        clean = _BASE64_IMG_RE.sub('', text).strip()
        words = _WORD_RE.findall(clean)
        if not words:
            return ""
        tail_words = words[-num_tokens:]
        idx = clean.rfind(tail_words[0]) if tail_words else -1
        if idx >= 0:
            return clean[idx:].strip()
        return ' '.join(tail_words)


# ============================================================================
# DOCUMENT PROCESSOR  (Public API)
# ============================================================================

class DocumentProcessor:
    """Public API — interface tương thích với main.py."""

    def __init__(
        self,
        use_gpu: bool = True,
        correction_model: str = "student",   # NEW: teacher / student / nano
        lang: str = "vi",                     # NEW: OCR language
    ):
        self.use_gpu = use_gpu
        from paddle_ocr_processor import get_paddle_ocr_processor
        logger.info("🚀 Initializing PaddleOCR v3.7.0 processor...")
        self.paddle_processor = get_paddle_ocr_processor(
            use_gpu=use_gpu,
            enable_spell_correction=True,
            enable_table_normalization=True,
            correction_model=correction_model,
            lang=lang,
        )
        logger.info("✅ PaddleOCR v3.7.0 processor ready")
        self.chunker = SmartChunker(
            target_chunk_size=450,
            min_chunk_size=80,
            max_chunk_size=700,
            overlap_size=80,
        )
        logger.info("✅ Smart Chunker v4 initialized")

    def process_pdf(self, file_path: str) -> str:
        logger.info("   🚀 Processing PDF (PaddleOCR v3.7.0 + PP-StructureV3)...")
        result = self.paddle_processor.process_pdf(file_path)
        if not result or len(result.strip()) < 50:
            raise ValueError(f"PaddleOCR returned empty result for PDF: {file_path}")
        logger.info(f"   ✅ PDF: {len(result)} chars")
        return result

    def process_word(self, file_path: str) -> str:
        logger.info("   🚀 Processing DOCX...")
        result = self.paddle_processor.process_docx(file_path)
        if not result or len(result.strip()) < 50:
            raise ValueError(f"PaddleOCR returned empty result for DOCX: {file_path}")
        logger.info(f"   ✅ DOCX: {len(result)} chars")
        return result

    def process_excel(self, file_path: str) -> str:
        logger.info("   🚀 Processing Excel...")
        result = self.paddle_processor._process_excel(file_path)
        if not result or len(result.strip()) < 20:
            raise ValueError(f"PaddleOCR returned empty result for Excel: {file_path}")
        logger.info(f"   ✅ Excel: {len(result)} chars")
        return result

    def process_image(self, file_path: str) -> str:
        logger.info("   🚀 Processing Image (PP-StructureV3 + PP-OCRv5)...")
        result = self.paddle_processor.process_image(file_path)
        if not result or len(result.strip()) < 10:
            raise ValueError(f"PaddleOCR returned empty result for image: {file_path}")
        logger.info(f"   ✅ Image: {len(result)} chars")
        return result

    def process_text(self, text_content: str) -> str:
        logger.info("   📝 Processing plain text (ProtonX correction)...")
        # Apply ProtonX correction to plain text
        result = self.paddle_processor._corrector.correct(text_content)
        result = self.paddle_processor._post_process(result)
        logger.info(f"   ✅ Text: {len(result)} chars")
        return result

    def parse_markdown_to_chunk_records(
        self,
        markdown_content: str,
        document_id: str,
        chunk_mode: str = "smart",
    ) -> List[ChunkRecord]:
        logger.info(f"🧠 Smart Chunking v4 | doc={document_id} | mode={chunk_mode}")
        chunker = SmartChunker(
            target_chunk_size=450,
            min_chunk_size=80,
            max_chunk_size=700,
            overlap_size=80,
            chunk_mode=chunk_mode,
        )
        records = chunker.chunk_document(markdown_content, document_id)

        from collections import Counter
        type_counts = Counter(r.chunk_type for r in records)
        logger.info(f"  Chunk types: {dict(type_counts)}")
        img_count = sum(1 for r in records if r.has_image)
        tbl_count = sum(1 for r in records if r.has_table)
        toc_count = type_counts.get("toc", 0)
        logger.info(
            f"  has_image={img_count}  has_table={tbl_count}  toc={toc_count}"
        )
        return records

    # Backward compat
    def parse_markdown_to_chunks(self, markdown_content: str) -> List[Dict]:
        records = self.parse_markdown_to_chunk_records(markdown_content, "unknown")
        return [{"content": r.content_with_ctx, **r.__dict__} for r in records]