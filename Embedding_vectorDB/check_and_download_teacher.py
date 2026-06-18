#!/usr/bin/env python3
"""
check_and_download_teacher.py
==============================
1. Kiểm tra model teacher có trong cache chưa
2. Nếu chưa → download (cần internet lần đầu)
3. Test correction thật để xác nhận model đang chạy đúng
4. Debug tại sao changed=whitespace_only (ProtonX offline mode vấn đề)

Usage:
    python check_and_download_teacher.py
    python check_and_download_teacher.py --download   # force download nếu chưa có
    python check_and_download_teacher.py --test-only  # chỉ test correction
"""

import os
import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

MODELS = {
    "teacher": "protonx-models/protonx-legal-tc",
    "student": "protonx-models/distilled-protonx-legal-tc",
    "nano":    "protonx-models/nano-protonx-legal-tc",
}

# Text tiếng Việt có lỗi chính tả rõ ràng để test
TEST_TEXTS = [
    "ngan hang chinh sach xa hoi ho tro nguoi dan vay von",
    "bo truong bo tai chinh ky quyet dinh phe duyet ke hoach",
    "cac don vi truc thuoc can thuc hien nghiem chinh quy dinh nay",
]


# ============================================================================
# 1. KIỂM TRA CACHE
# ============================================================================

def find_hf_cache_dirs() -> list:
    """Tìm tất cả thư mục cache HuggingFace có thể."""
    candidates = [
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.getenv("HF_HOME", ""),
        os.getenv("TRANSFORMERS_CACHE", ""),
        "/app/hf_cache",
    ]
    return [d for d in candidates if d and os.path.isdir(d)]


def check_model_in_cache(model_id: str) -> tuple[bool, str]:
    """Kiểm tra model có trong cache không. Trả về (found, path)."""
    # HuggingFace lưu theo pattern: models--{org}--{name}
    model_dir_name = "models--" + model_id.replace("/", "--")

    for cache_dir in find_hf_cache_dirs():
        model_path = os.path.join(cache_dir, model_dir_name)
        if os.path.isdir(model_path):
            # Kiểm tra có snapshots (files thật) không
            snapshots = os.path.join(model_path, "snapshots")
            if os.path.isdir(snapshots) and os.listdir(snapshots):
                return True, model_path

    return False, ""


def print_cache_status():
    logger.info("=" * 60)
    logger.info("📦  CACHE STATUS")
    logger.info("=" * 60)

    cache_dirs = find_hf_cache_dirs()
    logger.info(f"Cache dirs tìm thấy: {cache_dirs}")

    for name, model_id in MODELS.items():
        found, path = check_model_in_cache(model_id)
        status = f"✅ {path}" if found else "❌ chưa có"
        logger.info(f"  {name:8s} ({model_id}): {status}")

    logger.info("")


# ============================================================================
# 2. DOWNLOAD MODEL
# ============================================================================

def download_model(model_size: str = "teacher"):
    """Download model từ HuggingFace (cần internet)."""
    model_id = MODELS[model_size]
    logger.info(f"📥  Downloading: {model_id}")

    # Tắt offline mode để download
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if var in os.environ:
            del os.environ[var]
            logger.info(f"   Removed env: {var}")

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
        )
        logger.info(f"✅  Downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"❌  Download failed: {e}")
        logger.error("    Kiểm tra kết nối internet và quyền truy cập HuggingFace")
        return None


# ============================================================================
# 3. TEST CORRECTION THỰC TẾ
# ============================================================================

def test_protonx_correction(model_size: str = "teacher", force_online: bool = False):
    """
    Test ProtonX correction thực tế.
    Quan trọng: phải gọi TRƯỚC khi import paddle_ocr_processor
    để tránh HF_HUB_OFFLINE đã được set.
    """
    logger.info("=" * 60)
    logger.info(f"🧪  TEST PROTONX CORRECTION (model={model_size})")
    logger.info("=" * 60)

    # Kiểm tra offline mode
    hf_offline = os.getenv("HF_HUB_OFFLINE", "0")
    tf_offline  = os.getenv("TRANSFORMERS_OFFLINE", "0")
    logger.info(f"   HF_HUB_OFFLINE      = {hf_offline}")
    logger.info(f"   TRANSFORMERS_OFFLINE = {tf_offline}")

    if force_online:
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
            os.environ.pop(var, None)
        logger.info("   → Force online mode để test")

    try:
        from protonx import ProtonX

        # Thử cả 2 mode
        for mode in ("offline", "online") if not force_online else ("online",):
            logger.info(f"\n--- ProtonX(mode='{mode}') ---")
            try:
                client = ProtonX(mode=mode)
                model_name = MODELS[model_size]

                for i, text in enumerate(TEST_TEXTS, 1):
                    result = client.text.correct(input=text, top_k=1, model=model_name)
                    if result and "data" in result and result["data"]:
                        candidates = result["data"][0].get("candidates", [])
                        if candidates:
                            output = candidates[0]["output"]
                            changed = output.strip() != text.strip()
                            logger.info(f"  [{i}] Input : {text}")
                            logger.info(f"  [{i}] Output: {output}")
                            logger.info(f"  [{i}] Changed: {'YES ✅' if changed else 'NO (whitespace only?)'}")
                        else:
                            logger.warning(f"  [{i}] No candidates in response: {result}")
                    else:
                        logger.warning(f"  [{i}] Unexpected response: {result}")

                logger.info(f"✅  mode='{mode}' works")
                break  # nếu offline OK thì không cần thử online

            except Exception as e:
                logger.warning(f"❌  mode='{mode}' failed: {e}")

    except ImportError:
        logger.error("❌  protonx not installed. Run: pip install protonx")


# ============================================================================
# 4. PHÂN TÍCH VẤN ĐỀ whitespace_only
# ============================================================================

def analyze_whitespace_only_issue():
    """
    Giải thích tại sao ProtonX trả về changed=whitespace_only
    và cách fix.
    """
    logger.info("=" * 60)
    logger.info("🔍  PHÂN TÍCH: Tại sao changed=whitespace_only?")
    logger.info("=" * 60)
    logger.info("""
Nguyên nhân có thể:

1. ProtonX offline mode không tìm thấy model weights trong cache
   → Dùng fallback: chỉ normalize whitespace, không sửa chính tả
   → Fix: Download model trước (chạy với --download)

2. Model đang chạy đúng nhưng text từ pdfplumber đã chuẩn
   → PDF là text gốc (không phải scan) → không có lỗi chính tả
   → Đây là hành vi ĐÚNG, không phải lỗi

3. HF_HUB_OFFLINE=1 được set TRƯỚC khi import protonx
   → protonx không thể resolve model config dù cache có sẵn
   → Fix: Xem bên dưới

CÁCH XÁC NHẬN:
   Chạy: python check_and_download_teacher.py --test-only
   Nếu text TEST_TEXTS được sửa đúng → model chạy OK, PDF text đã chuẩn
   Nếu TEST_TEXTS KHÔNG được sửa     → model có vấn đề
""")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="teacher",
                        choices=["teacher", "student", "nano"])
    parser.add_argument("--download",   action="store_true",
                        help="Download model nếu chưa có trong cache")
    parser.add_argument("--test-only",  action="store_true",
                        help="Chỉ test correction, bỏ qua check cache")
    parser.add_argument("--force-online", action="store_true",
                        help="Force online mode khi test (tắt HF_HUB_OFFLINE)")
    args = parser.parse_args()

    if not args.test_only:
        print_cache_status()
        analyze_whitespace_only_issue()

        found, path = check_model_in_cache(MODELS[args.model])
        if not found:
            logger.warning(f"⚠️  Model '{args.model}' CHƯA có trong cache!")
            if args.download:
                download_model(args.model)
            else:
                logger.info("   Chạy lại với --download để tải về")
                logger.info(f"   Hoặc set: PROTONX_ALLOW_ONLINE_WARMUP=1 trước khi start main.py")
        else:
            logger.info(f"✅  Model '{args.model}' đã có trong cache: {path}")

    test_protonx_correction(args.model, force_online=args.force_online)


if __name__ == "__main__":
    main()