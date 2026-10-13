# media_store.py - Dual-Layer Media Storage (Local Disk & Cloud Supabase Storage)
import os
import sys
import hashlib
import base64
import mimetypes
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add root directory to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import config


def get_mime_type(filename: str) -> str:
    """Infers MIME type from filename with sensible defaults for images."""
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        return mime
    ext = Path(filename).suffix.lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".svg": "image/svg+xml",
        ".gif": "image/gif"
    }
    return mapping.get(ext, "image/png")


def save_media(
    media_bytes: bytes,
    original_filename: str = "image.png",
    mime_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Saves media payload:
    1. Writes to local disk cache (db/media/<hash>_<filename>).
    2. If Supabase is enabled, attempts background/cloud storage upload.
    3. Returns metadata dict with local path and web accessible URL.
    """
    if not media_bytes:
        raise ValueError("Media bytes cannot be empty.")

    # Generate content-addressed hash to deduplicate identical images
    content_hash = hashlib.sha256(media_bytes).hexdigest()[:12]
    clean_name = Path(original_filename).name.replace(" ", "_")
    stored_filename = f"{content_hash}_{clean_name}"

    resolved_mime = mime_type or get_mime_type(original_filename)

    # 1. Local Storage (always write to disk cache)
    config.LOCAL_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    local_path = config.LOCAL_MEDIA_DIR / stored_filename
    if not local_path.exists():
        with open(local_path, "wb") as f:
            f.write(media_bytes)

    # Web URL endpoint served by GUI FastAPI server
    media_url = f"/media/{stored_filename}"

    # 2. Cloud Supabase Storage Upload (Optional / Best Effort)
    if config.USE_SUPABASE:
        try:
            from supabase import create_client
            supabase = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
            bucket_name = config.SUPABASE_MEDIA_BUCKET

            # Check if bucket exists, create if missing
            try:
                existing_buckets = [b.name for b in supabase.storage.list_buckets()]
                if bucket_name not in existing_buckets:
                    supabase.storage.create_bucket(bucket_name, options={"public": True})
            except Exception:
                pass  # Ignore bucket check errors; proceed to upload

            # Upload to bucket
            supabase.storage.from_(bucket_name).upload(
                path=stored_filename,
                file=media_bytes,
                file_options={"content-type": resolved_mime, "upsert": "true"}
            )
            # Use public cloud URL if available
            cloud_url = supabase.storage.from_(bucket_name).get_public_url(stored_filename)
            if cloud_url:
                media_url = cloud_url
        except Exception as e:
            # Gracefully fallback to local media serving
            print(f"[MediaStore] Supabase storage upload skipped/fallback: {e}")

    return {
        "filename": stored_filename,
        "local_path": str(local_path),
        "url": media_url,
        "mime_type": resolved_mime,
        "size_bytes": len(media_bytes)
    }


def load_media_base64(file_path_or_filename: str) -> Tuple[str, str]:
    """
    Loads media file from disk and converts to Base64 in memory for Gemini Vision.
    Returns: (base64_string, mime_type)
    """
    p = Path(file_path_or_filename)
    if not p.is_absolute():
        p = config.LOCAL_MEDIA_DIR / p.name

    if not p.exists():
        raise FileNotFoundError(f"Media file not found: {p}")

    mime_type = get_mime_type(p.name)
    with open(p, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return encoded, mime_type


def get_media_url(filename: str) -> str:
    """Returns the local web-accessible URL for a media file."""
    clean_name = Path(filename).name
    return f"/media/{clean_name}"
