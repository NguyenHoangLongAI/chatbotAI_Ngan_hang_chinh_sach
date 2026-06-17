#!/usr/bin/env python3
"""
Batch Upload Script - Xử lý tài liệu từ thư mục Tai_lieu_training
Upload lên document-api: POST /api/v1/process-document

Usage:
    python batch_upload.py
    python batch_upload.py --dir Embedding_vectorDB/Tai_lieu_training
    python batch_upload.py --url http://localhost:8000 --dir ./docs --workers 2
"""

import os
import re
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("batch_upload.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt",
                        ".png", ".jpg", ".jpeg", ".tiff"}

CONTENT_TYPES = {
    ".pdf":  "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc":  "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls":  "application/vnd.ms-excel",
    ".txt":  "text/plain",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def sanitize_id(text: str) -> str:
    """Chuyển tên file thành document_id hợp lệ."""
    sanitized = re.sub(r"[^\w\-_.]", "_", text)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")[:90]          # Milvus VARCHAR max 90


def collect_files(directory: str) -> list[Path]:
    """Lấy tất cả file hỗ trợ trong thư mục (recursive)."""
    root = Path(directory)
    if not root.exists():
        logger.error(f"Thư mục không tồn tại: {directory}")
        sys.exit(1)

    files = [
        p for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def check_api_health(base_url: str, timeout: int = 10) -> bool:
    """Kiểm tra API có sẵn sàng không."""
    try:
        resp = requests.get(f"{base_url}/api/v1/health", timeout=timeout)
        data = resp.json()
        status = data.get("status", "")
        if status == "healthy":
            logger.info(f"✅  API sẵn sàng: {base_url}  |  {data}")
            return True
        logger.warning(f"⚠️   API trạng thái không healthy: {data}")
        return True   # vẫn thử upload
    except Exception as e:
        logger.error(f"❌  Không kết nối được API {base_url}: {e}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Upload single file
# ──────────────────────────────────────────────────────────────────────────────

def upload_file(
    file_path: Path,
    base_url: str,
    chunk_mode: str = "smart",
    timeout: int = 600,
) -> dict:
    """
    Upload một file lên /api/v1/process-document.

    Returns dict với keys: file, status, document_id, detail, elapsed
    """
    start = time.time()
    ext = file_path.suffix.lower()
    document_id = sanitize_id(file_path.stem)
    content_type = CONTENT_TYPES.get(ext, "application/octet-stream")

    result = {
        "file": str(file_path),
        "document_id": document_id,
        "status": "FAILED",
        "detail": "",
        "elapsed": 0.0,
    }

    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, content_type)}
            data  = {"document_id": document_id, "chunk_mode": chunk_mode}

            resp = requests.post(
                f"{base_url}/api/v1/process-document",
                files=files,
                data=data,
                timeout=timeout,
            )

        elapsed = round(time.time() - start, 1)
        result["elapsed"] = elapsed

        if resp.status_code == 200:
            body = resp.json()
            stats = body.get("processing_stats", {})
            storage = body.get("storage", {})
            result["status"] = "SUCCESS"
            result["detail"] = (
                f"chunks={stats.get('total_chunks', '?')} | "
                f"vectors={stats.get('stored_embeddings', '?')} | "
                f"url={storage.get('public_url', 'N/A')}"
            )
        else:
            result["detail"] = f"HTTP {resp.status_code}: {resp.text[:300]}"

    except requests.exceptions.Timeout:
        result["elapsed"] = round(time.time() - start, 1)
        result["detail"] = f"Timeout sau {timeout}s"
    except Exception as e:
        result["elapsed"] = round(time.time() - start, 1)
        result["detail"] = str(e)

    icon = "✅" if result["status"] == "SUCCESS" else "❌"
    logger.info(
        f"{icon}  [{result['status']:7s}]  {file_path.name:<40s}  "
        f"{result['elapsed']:6.1f}s  {result['detail']}"
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch upload tài liệu lên document-api"
    )
    parser.add_argument(
        "--dir",
        default="/mnt/data/nhlong22/chatbotAI_Ngan_hang_chinh_sach/Embedding_vectorDB/Tai_lieu_training",
        help="Thư mục chứa tài liệu (default: Embedding_vectorDB/Tai_lieu_training)",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8022",
        help="Base URL của document-api (default: http://localhost:8022)",
    )
    parser.add_argument(
        "--chunk-mode",
        default="smart",
        choices=["smart", "sentence", "legacy"],
        help="Chế độ chunking (default: smart)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Số luồng upload song song (default: 1, khuyến nghị ≤ 3)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout mỗi file (giây, default: 300)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay giữa các lần upload tuần tự (giây, default: 1.0)",
    )
    parser.add_argument(
        "--output",
        default="batch_upload_results.json",
        help="File JSON lưu kết quả (default: batch_upload_results.json)",
    )
    args = parser.parse_args()

    # ── Kiểm tra API ──────────────────────────────────────────────────────────
    if not check_api_health(args.url):
        logger.error("Dừng: API không phản hồi.")
        sys.exit(1)

    # ── Lấy danh sách file ───────────────────────────────────────────────────
    files = collect_files(args.dir)
    if not files:
        logger.warning(f"Không tìm thấy file nào trong: {args.dir}")
        sys.exit(0)

    logger.info("=" * 70)
    logger.info(f"📁  Thư mục : {args.dir}")
    logger.info(f"🌐  API URL : {args.url}")
    logger.info(f"📦  Chunk   : {args.chunk_mode}")
    logger.info(f"⚙️   Workers : {args.workers}")
    logger.info(f"📄  Tổng file: {len(files)}")
    logger.info("=" * 70)

    # ── Upload ────────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    t0 = time.time()

    if args.workers <= 1:
        # Tuần tự (an toàn nhất với model lớn)
        for i, fp in enumerate(files, 1):
            logger.info(f"[{i:3d}/{len(files)}] {fp.name}")
            result = upload_file(fp, args.url, args.chunk_mode, args.timeout)
            all_results.append(result)
            if i < len(files):
                time.sleep(args.delay)
    else:
        # Song song
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(upload_file, fp, args.url, args.chunk_mode, args.timeout): fp
                for fp in files
            }
            for fut in as_completed(futures):
                all_results.append(fut.result())

    total_elapsed = round(time.time() - t0, 1)

    # ── Thống kê ──────────────────────────────────────────────────────────────
    success = [r for r in all_results if r["status"] == "SUCCESS"]
    failed  = [r for r in all_results if r["status"] != "SUCCESS"]

    logger.info("=" * 70)
    logger.info(f"🏁  Hoàn thành  |  Tổng: {len(all_results)}  |  "
                f"✅ {len(success)}  |  ❌ {len(failed)}  |  "
                f"Thời gian: {total_elapsed}s")

    if failed:
        logger.warning("── File lỗi ──────────────────────────────────────────")
        for r in failed:
            logger.warning(f"  ❌  {Path(r['file']).name}: {r['detail']}")

    # ── Lưu kết quả JSON ─────────────────────────────────────────────────────
    output_data = {
        "summary": {
            "total":   len(all_results),
            "success": len(success),
            "failed":  len(failed),
            "elapsed": total_elapsed,
            "directory": args.dir,
            "api_url":   args.url,
            "chunk_mode": args.chunk_mode,
        },
        "results": all_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"💾  Kết quả lưu tại: {args.output}")
    logger.info("=" * 70)

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()