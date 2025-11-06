"""Upload images and audio files to MinIO and update compound documents."""

import os
import json
from pathlib import Path
from typing import Dict, Any, List

from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

load_dotenv(override=True)


# MinIO configuration
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"

# Bucket names
IMAGES_BUCKET = "chemistry-images"
AUDIO_BUCKET = "chemistry-audio"

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "src" / "data"


def init_minio_client() -> Minio:
    """Initialize MinIO client.

    Returns:
        Minio client instance
    """
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE
    )


def upload_file(client: Minio, bucket_name: str, local_path: Path, object_name: str) -> str:
    """Upload file to MinIO.

    Args:
        client: MinIO client
        bucket_name: Target bucket name
        local_path: Local file path
        object_name: Object name in bucket

    Returns:
        Public URL to the uploaded file
    """
    try:
        # Determine content type
        content_type = "application/octet-stream"
        suffix = local_path.suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg"]:
            content_type = f"image/{suffix[1:]}"
        elif suffix == ".wav":
            content_type = "audio/wav"

        # Upload file
        client.fput_object(
            bucket_name,
            object_name,
            str(local_path),
            content_type=content_type
        )

        # Generate public URL
        url = f"http://{MINIO_ENDPOINT}/{bucket_name}/{object_name}"
        print(f"  ✓ Uploaded: {object_name} -> {url}")
        return url

    except S3Error as e:
        print(f"  ✗ Error uploading {object_name}: {e}")
        return ""


def process_compounds(client: Minio) -> List[Dict[str, Any]]:
    """Upload files and update compound documents with MinIO URLs.

    Args:
        client: MinIO client

    Returns:
        Updated list of compound documents
    """
    compounds_file = DATA_DIR / "compounds.json"

    if not compounds_file.exists():
        print(f"✗ Compounds file not found: {compounds_file}")
        return []

    # Load compounds
    with open(compounds_file, "r", encoding="utf-8") as f:
        compounds = json.load(f)

    print(f"\nProcessing {len(compounds)} compounds...")

    updated_compounds = []

    for compound in compounds:
        doc_id = compound.get("doc_id", "unknown")
        print(f"\n[{doc_id}]")

        # Upload image if exists
        image_path = compound.get("image_path", "")
        if image_path:
            local_image = PROJECT_ROOT / image_path
            if local_image.exists():
                object_name = f"{doc_id}{local_image.suffix}"
                image_url = upload_file(client, IMAGES_BUCKET, local_image, object_name)
                compound["image_url"] = image_url
            else:
                print(f"  ✗ Image not found: {local_image}")
                compound["image_url"] = ""
        else:
            compound["image_url"] = ""

        # Upload audio if exists
        audio_path = compound.get("audio_path", "")
        if audio_path:
            local_audio = PROJECT_ROOT / audio_path
            if local_audio.exists():
                object_name = f"{doc_id}{local_audio.suffix}"
                audio_url = upload_file(client, AUDIO_BUCKET, local_audio, object_name)
                compound["audio_url"] = audio_url
            else:
                print(f"  ✗ Audio not found: {local_audio}")
                compound["audio_url"] = ""
        else:
            compound["audio_url"] = ""

        updated_compounds.append(compound)

    return updated_compounds


def save_updated_compounds(compounds: List[Dict[str, Any]]):
    """Save updated compounds to JSON file.

    Args:
        compounds: List of updated compound documents
    """
    compounds_file = DATA_DIR / "compounds.json"

    with open(compounds_file, "w", encoding="utf-8") as f:
        json.dump(compounds, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Updated compounds saved to: {compounds_file}")


def main():
    """Main upload pipeline."""
    print("=" * 80)
    print("Upload Files to MinIO")
    print("=" * 80)

    # Initialize MinIO client
    print("\nConnecting to MinIO...")
    try:
        client = init_minio_client()
        # Test connection
        client.list_buckets()
        print(f"✓ Connected to MinIO at {MINIO_ENDPOINT}")
    except Exception as e:
        print(f"✗ Failed to connect to MinIO: {e}")
        print("\nMake sure MinIO is running:")
        print("  docker compose up -d minio")
        return

    # Process compounds
    print("\nUploading files...")
    updated_compounds = process_compounds(client)

    if not updated_compounds:
        print("\n✗ No compounds processed")
        return

    # Save updated compounds
    print("\nSaving updated compounds...")
    save_updated_compounds(updated_compounds)

    print("\n" + "=" * 80)
    print("✓ Upload completed successfully!")
    print("=" * 80)
    print(f"\nMinIO Console: http://{MINIO_ENDPOINT.split(':')[0]}:9001")
    print(f"  Username: {MINIO_ACCESS_KEY}")
    print(f"  Password: {MINIO_SECRET_KEY}")


if __name__ == "__main__":
    main()
