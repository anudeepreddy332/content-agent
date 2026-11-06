"""Shadow Qdrant collection naming and schema pins."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PRODUCTION_INDEX_DIR = ROOT / "kb/indexes/cswp_v1"
SHADOW_MANIFEST_FILE = "qdrant_shadow_manifest.json"
PRODUCTION_COLLECTION = "machinist_evergreen"
SERVING_ALIAS = "kb_cswp_serving"

# Deterministic UUID namespace for point IDs (uuid5 over canonical chunk_id).
POINT_ID_NAMESPACE = uuid.UUID("ca000000-0000-5000-8000-000000000005")

PAYLOAD_SCHEMA_VERSION = "cswp_shadow_payload_v1"
QDRANT_IMAGE = "qdrant/qdrant:v1.9.2"
VECTOR_SIZE = 384

BLOCKED_COLLECTIONS = frozenset({PRODUCTION_COLLECTION})


def shadow_collection_name(index_digest16: str) -> str:
    return f"ca_cswp_v1_{index_digest16}"


def shadow_qdrant_url() -> str:
    return os.getenv("SHADOW_QDRANT_URL", os.getenv("QDRANT_URL", "http://localhost:6333"))
