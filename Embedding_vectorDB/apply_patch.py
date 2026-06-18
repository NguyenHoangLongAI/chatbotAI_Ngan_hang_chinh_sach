#!/usr/bin/env python3
"""
apply_patch.py
==============
Tự động chèn 3 hàm helper bị thiếu vào paddle_ocr_processor.py:
  - _paddle_actual_device()
  - _is_whitespace_only_change()
  - _diff_summary()

Usage:
    python apply_patch.py
    python apply_patch.py --file /path/to/paddle_ocr_processor.py
"""

import re
import sys
import shutil
import argparse
from pathlib import Path

# ── Code cần chèn vào ────────────────────────────────────────────────────────

PATCH_CODE = '''

# ============================================================================
# MODULE-LEVEL HELPERS  (required by PaddleOCREngine + VietnameseTextCorrector)
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
    """
    True nếu sự khác biệt giữa original và corrected chỉ là whitespace
    (khoảng trắng, newline, tab) — nội dung thực sự giống nhau.
    """
    import re as _re
    return _re.sub(r"\\s+", "", original) == _re.sub(r"\\s+", "", corrected)


def _diff_summary(original: str, corrected: str, max_examples: int = 3) -> str:
    """Tóm tắt ngắn gọn những thay đổi từ ngữ giữa original và corrected."""
    import difflib
    orig_words = original.split()
    corr_words = corrected.split()
    matcher    = difflib.SequenceMatcher(None, orig_words, corr_words)
    examples   = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert", "delete") and len(examples) < max_examples:
            orig_seg = " ".join(orig_words[i1:i2])[:60]
            corr_seg = " ".join(corr_words[j1:j2])[:60]
            examples.append(f"  [{len(examples) + 1}] {orig_seg!r} → {corr_seg!r}")
    return ("\\n" + "\\n".join(examples)) if examples else " (no word-level diff)"

'''

# ── Anchor: chèn SAU dòng "logger = logging.getLogger(__name__)" ─────────────

ANCHOR_PATTERN = re.compile(
    r"(logger\s*=\s*logging\.getLogger\(__name__\))",
    re.MULTILINE,
)

ALREADY_PATCHED_MARKER = "_paddle_actual_device"


def patch_file(target: Path) -> None:
    if not target.exists():
        print(f"❌  File không tồn tại: {target}")
        sys.exit(1)

    content = target.read_text(encoding="utf-8")

    # Kiểm tra đã patch chưa
    if ALREADY_PATCHED_MARKER in content:
        print(f"ℹ️   File đã chứa '{ALREADY_PATCHED_MARKER}' — có thể đã được patch rồi.")
        answer = input("Tiếp tục patch lại? [y/N]: ").strip().lower()
        if answer != "y":
            print("Hủy.")
            sys.exit(0)

    # Tìm anchor
    m = ANCHOR_PATTERN.search(content)
    if not m:
        print("❌  Không tìm thấy anchor 'logger = logging.getLogger(__name__)'.")
        print("    Hãy chèn thủ công — xem paddle_ocr_processor_patch.py để biết chi tiết.")
        sys.exit(1)

    # Backup
    backup = target.with_suffix(".py.bak")
    shutil.copy2(target, backup)
    print(f"💾  Backup → {backup}")

    # Chèn patch SAU anchor
    insert_pos = m.end()
    new_content = content[:insert_pos] + PATCH_CODE + content[insert_pos:]

    target.write_text(new_content, encoding="utf-8")
    print(f"✅  Đã patch thành công: {target}")
    print(f"    Đã thêm: _paddle_actual_device, _is_whitespace_only_change, _diff_summary")


def main():
    parser = argparse.ArgumentParser(description="Auto-patch paddle_ocr_processor.py")
    parser.add_argument(
        "--file",
        default="paddle_ocr_processor.py",
        help="Đường dẫn tới paddle_ocr_processor.py (default: Embedding_vectorDB/paddle_ocr_processor.py)",
    )
    args = parser.parse_args()

    target = Path(args.file)
    print(f"🔧  Đang patch: {target.resolve()}")
    patch_file(target)


if __name__ == "__main__":
    main()