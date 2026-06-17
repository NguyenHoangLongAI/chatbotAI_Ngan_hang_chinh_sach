#!/usr/bin/env python3
"""
Simple URL Uploader - Upload files và lưu URLs với filename embedding

Chỉ làm 2 việc:
1. Upload files lên public storage
2. Lưu URLs vào document_urls collection (với filename embedding)

Usage:
    python upload_and_store_urls_simple.py --input van_ban_downloads --storage minio
"""

import os
import re
import time
import argparse
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleURLUploader:
    """
    Upload files và lưu URLs (KHÔNG process, KHÔNG embed document)
    """

    def __init__(
            self,
            input_dir: str,
            storage_type: str = "minio",
            milvus_host: str = "localhost",
            milvus_port: str = "19530",
            storage_config: Dict = None
    ):
        self.input_dir = input_dir
        self.supported_ext = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".txt")

        # Initialize uploader
        self.uploader = self._init_uploader(storage_type, storage_config or {})

        # Initialize document_urls manager
        from document_urls_collection import DocumentURLsManager
        self.urls_manager = DocumentURLsManager(
            host=milvus_host,
            port=milvus_port
        )

        # Ensure collection exists
        self.urls_manager.create_collection()
        logger.info("✅ Document URLs collection ready")

    def _init_uploader(self, storage_type: str, config: Dict):
        """Initialize storage uploader"""

        if storage_type == "minio":
            try:
                from minio import Minio

                endpoint = config.get("endpoint", "localhost:9000")
                access_key = config.get("access_key", "minioadmin")
                secret_key = config.get("secret_key", "minioadmin")
                bucket_name = config.get("bucket_name", "public-documents")
                secure = config.get("secure", False)

                logger.info(f"🔧 Connecting to MinIO: {endpoint}")

                client = Minio(
                    endpoint,
                    access_key=access_key,
                    secret_key=secret_key,
                    secure=secure
                )

                # Create bucket if not exists
                if not client.bucket_exists(bucket_name):
                    client.make_bucket(bucket_name)
                    logger.info(f"✅ Created bucket: {bucket_name}")

                # Set public read policy
                import json
                policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Effect": "Allow",
                            "Principal": {"AWS": "*"},
                            "Action": ["s3:GetObject"],
                            "Resource": [f"arn:aws:s3:::{bucket_name}/*"]
                        }
                    ]
                }

                client.set_bucket_policy(bucket_name, json.dumps(policy))
                logger.info(f"✅ Bucket is public: {bucket_name}")

                return MinIOUploader(client, endpoint, bucket_name, secure)

            except ImportError:
                raise ImportError("MinIO library not installed. Run: pip install minio")

        elif storage_type == "cloudinary":
            try:
                import cloudinary
                import cloudinary.uploader

                if not all(k in config for k in ["cloud_name", "api_key", "api_secret"]):
                    raise ValueError("Cloudinary requires: cloud_name, api_key, api_secret")

                cloudinary.config(
                    cloud_name=config["cloud_name"],
                    api_key=config["api_key"],
                    api_secret=config["api_secret"]
                )

                logger.info(f"✅ Cloudinary connected: {config['cloud_name']}")
                return CloudinaryUploader(cloudinary)

            except ImportError:
                raise ImportError("Cloudinary library not installed. Run: pip install cloudinary")

        elif storage_type == "github":
            try:
                from github import Github

                if not all(k in config for k in ["repo_name", "access_token"]):
                    raise ValueError("GitHub requires: repo_name, access_token")

                github = Github(config["access_token"])
                repo = github.get_repo(config["repo_name"])
                tag = config.get("release_tag", "documents-v1.0")

                # Get or create release
                try:
                    release = repo.get_release(tag)
                except:
                    release = repo.create_git_release(
                        tag=tag,
                        name=f"Documents {tag}",
                        message="Public document storage"
                    )

                logger.info(f"✅ GitHub Release ready: {config['repo_name']}/{tag}")
                return GitHubUploader(release)

            except ImportError:
                raise ImportError("PyGithub not installed. Run: pip install PyGithub")

        else:
            raise ValueError(f"Unknown storage: {storage_type}")

    def sanitize_id(self, text: str) -> str:
        """Sanitize document ID"""
        sanitized = re.sub(r"[^\w\-_.]", "_", text)
        sanitized = re.sub(r"_+", "_", sanitized)
        return sanitized.strip("_")

    def process_file(self, file_path: str) -> Dict:
        """
        Upload file và lưu URL

        Returns:
            Dict with status and results
        """
        filename = os.path.basename(file_path)
        # Use filename without extension as document_id (match với embedding collection)
        document_id = self.sanitize_id(os.path.splitext(filename)[0])
        file_ext = Path(file_path).suffix.lower()

        result = {
            "filename": filename,
            "document_id": document_id,
            "status": "FAILED",
            "error": None
        }

        try:
            # Step 1: Upload file
            logger.info(f"📤 [1/2] Uploading: {filename}")
            public_url = self.uploader.upload(file_path, document_id)

            if not public_url:
                raise Exception("Upload returned no URL")

            logger.info(f"✅ [1/2] Uploaded: {public_url[:60]}...")
            result["public_url"] = public_url

            # Step 2: Store URL with filename embedding
            logger.info(f"💾 [2/2] Storing URL in Milvus: {document_id}")

            url_stored = self.urls_manager.insert_url(
                document_id=document_id,
                url=public_url,
                filename=filename,
                file_type=file_ext
            )

            if not url_stored:
                raise Exception("URL storage failed")

            logger.info(f"✅ [2/2] URL stored with filename embedding")

            result["status"] = "SUCCESS"

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"❌ Failed: {e}")

        return result

    def run(self):
        """Process all files in directory"""

        # Validate input
        if not os.path.exists(self.input_dir):
            logger.error(f"❌ Directory not found: {self.input_dir}")
            return None

        # Find files
        files = [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(self.supported_ext)
        ]

        if not files:
            logger.warning(f"⚠️ No supported files in {self.input_dir}")
            logger.info(f"   Supported: {self.supported_ext}")
            return None

        logger.info("=" * 60)
        logger.info("🚀 URL UPLOADER")
        logger.info("=" * 60)
        logger.info(f"📁 Input: {self.input_dir}")
        logger.info(f"📝 Files: {len(files)}")
        logger.info("=" * 60)

        results = {"success": [], "failed": []}

        for idx, filename in enumerate(files, 1):
            file_path = os.path.join(self.input_dir, filename)

            logger.info(f"\n[{idx}/{len(files)}] 📄 {filename}")

            result = self.process_file(file_path)

            if result["status"] == "SUCCESS":
                results["success"].append(result)
                logger.info(f"✅ [{idx}/{len(files)}] SUCCESS")
            else:
                results["failed"].append(result)
                logger.error(f"❌ [{idx}/{len(files)}] FAILED: {result['error']}")

            # Rate limiting
            if idx < len(files):
                time.sleep(0.5)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Success: {len(results['success'])}")
        logger.info(f"❌ Failed: {len(results['failed'])}")

        if results['success']:
            logger.info("\n✅ Uploaded documents:")
            for r in results['success']:
                logger.info(f"  - {r['document_id']}: {r['public_url'][:50]}...")

        if results['failed']:
            logger.info("\n❌ Failed documents:")
            for r in results['failed']:
                logger.info(f"  - {r['filename']}: {r['error']}")

        logger.info("=" * 60)

        return results


# ============================================================================
# Storage Uploader Classes
# ============================================================================

class MinIOUploader:
    """MinIO uploader"""

    def __init__(self, client, endpoint, bucket_name, secure):
        self.client = client
        self.endpoint = endpoint
        self.bucket_name = bucket_name
        self.secure = secure

    def upload(self, file_path: str, document_id: str) -> str:
        """Upload to MinIO and return public URL"""
        file_ext = Path(file_path).suffix.lower()
        object_name = f"{document_id}{file_ext}"

        # Content type mapping
        content_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.txt': 'text/plain'
        }

        content_type = content_types.get(file_ext, 'application/octet-stream')

        # Upload
        self.client.fput_object(
            self.bucket_name,
            object_name,
            file_path,
            content_type=content_type
        )

        # Generate public URL
        protocol = "https" if self.secure else "http"
        return f"{protocol}://{self.endpoint}/{self.bucket_name}/{object_name}"


class CloudinaryUploader:
    """Cloudinary uploader"""

    def __init__(self, cloudinary):
        self.cloudinary = cloudinary

    def upload(self, file_path: str, document_id: str) -> str:
        """Upload to Cloudinary and return public URL"""
        response = self.cloudinary.uploader.upload(
            file_path,
            public_id=document_id,
            resource_type="raw",
            folder="documents"
        )
        return response.get('secure_url')


class GitHubUploader:
    """GitHub Releases uploader"""

    def __init__(self, release):
        self.release = release

    def upload(self, file_path: str, document_id: str) -> str:
        """Upload to GitHub release and return public URL"""
        file_ext = Path(file_path).suffix
        asset_name = f"{document_id}{file_ext}"

        # Delete existing asset
        for asset in self.release.get_assets():
            if asset.name == asset_name:
                asset.delete_asset()

        # Upload new asset
        asset = self.release.upload_asset(
            file_path,
            name=asset_name
        )

        return asset.browser_download_url


# ============================================================================
# Main
# ============================================================================

def main():
    """Main with CLI"""

    parser = argparse.ArgumentParser(
        description="Upload files và lưu URLs vào Milvus",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required
    parser.add_argument(
        "--input",
        required=True,
        help="Input directory (required)"
    )

    parser.add_argument(
        "--storage",
        choices=["minio", "cloudinary", "github"],
        default="minio",
        help="Storage provider (default: minio)"
    )

    # MinIO
    parser.add_argument("--minio-endpoint", default="localhost:9000")
    parser.add_argument("--minio-access-key", default="minioadmin")
    parser.add_argument("--minio-secret-key", default="minioadmin")
    parser.add_argument("--minio-bucket", default="public-documents")

    # Cloudinary
    parser.add_argument("--cloudinary-cloud-name", default="")
    parser.add_argument("--cloudinary-api-key", default="")
    parser.add_argument("--cloudinary-api-secret", default="")

    # GitHub
    parser.add_argument("--github-repo", default="")
    parser.add_argument("--github-token", default="")
    parser.add_argument("--github-tag", default="documents-v1.0")

    # Milvus
    parser.add_argument("--milvus-host", default="localhost")
    parser.add_argument("--milvus-port", default="19530")

    args = parser.parse_args()

    # Build storage config
    storage_config = {}

    if args.storage == "minio":
        storage_config = {
            "endpoint": args.minio_endpoint,
            "access_key": args.minio_access_key,
            "secret_key": args.minio_secret_key,
            "bucket_name": args.minio_bucket,
            "secure": False
        }

    elif args.storage == "cloudinary":
        if not all([args.cloudinary_cloud_name, args.cloudinary_api_key, args.cloudinary_api_secret]):
            print("❌ Cloudinary requires: --cloudinary-cloud-name, --cloudinary-api-key, --cloudinary-api-secret")
            return 1

        storage_config = {
            "cloud_name": args.cloudinary_cloud_name,
            "api_key": args.cloudinary_api_key,
            "api_secret": args.cloudinary_api_secret
        }

    elif args.storage == "github":
        if not all([args.github_repo, args.github_token]):
            print("❌ GitHub requires: --github-repo, --github-token")
            return 1

        storage_config = {
            "repo_name": args.github_repo,
            "access_token": args.github_token,
            "release_tag": args.github_tag
        }

    try:
        # Run uploader
        uploader = SimpleURLUploader(
            input_dir=args.input,
            storage_type=args.storage,
            milvus_host=args.milvus_host,
            milvus_port=args.milvus_port,
            storage_config=storage_config
        )

        results = uploader.run()

        if results is None:
            return 1

        # Exit code
        if len(results['failed']) == 0:
            return 0
        elif len(results['success']) > 0:
            return 2  # Partial
        else:
            return 1  # All failed

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())