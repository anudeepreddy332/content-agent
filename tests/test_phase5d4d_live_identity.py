"""Phase 5D4D: live Qdrant payload identity and topology fail-closed gates."""

from __future__ import annotations

import copy

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

from agent.cswp.constants import PRODUCTION_INDEX_DIR
from agent.kb_backend.qdrant_serving import (
    QdrantServingError,
    clear_serving_cache,
    create_serving_alias,
    validate_startup,
)
from agent.retrieval.encoder import MODEL_REVISION, resolve_local_model_snapshot
from agent.shadow_qdrant.identity import collection_content_fingerprint
from agent.shadow_qdrant.index import full_rebuild, point_id_for_chunk


@pytest.fixture(scope="module")
def minilm_ready():
    resolve_local_model_snapshot()
    return MODEL_REVISION


@pytest.fixture
def live_collection(minilm_ready, tmp_path, monkeypatch):
    client = QdrantClient(":memory:")
    index_dir = tmp_path / "cswp_v1"
    index_dir.mkdir()
    for name in ("manifest.json", "units.jsonl"):
        source = PRODUCTION_INDEX_DIR / name
        (index_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    manifest = full_rebuild(client, index_dir=index_dir)
    create_serving_alias(client, manifest["collection_name"])
    monkeypatch.setattr("agent.kb_backend.qdrant_serving._serving_client", lambda: client)
    clear_serving_cache()
    return client, manifest


def _payloads(client, collection):
    points, _offset = client.scroll(
        collection_name=collection, limit=512, with_payload=True, with_vectors=False
    )
    return [copy.deepcopy(point.payload) for point in points]


def _assert_startup_rejects(client, manifest):
    clear_serving_cache()
    with pytest.raises(QdrantServingError):
        validate_startup()


def test_live_fingerprint_is_scroll_order_independent_and_exposed(live_collection):
    client, manifest = live_collection
    payloads = _payloads(client, manifest["collection_name"])
    first = collection_content_fingerprint(
        payloads,
        representation_fingerprint=manifest["representation_fingerprint"],
        compiler_version=manifest["cswp_compiler_version"],
    )
    second = collection_content_fingerprint(
        list(reversed(payloads)),
        representation_fingerprint=manifest["representation_fingerprint"],
        compiler_version=manifest["cswp_compiler_version"],
    )
    startup = validate_startup()
    assert first == second == manifest["index_fingerprint"]
    assert startup["live_collection_fingerprint"] == manifest["index_fingerprint"]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("retrieval_text", "corrupted serving text"),
        ("source_spans", [{"source_char_start": 1, "source_char_end": 2}]),
        ("chunk_id", "ca:cswp:corrupted-chunk-id"),
        ("next_chunk_id", "ca:cswp:corrupted-neighbor"),
    ],
)
def test_altered_serving_payload_fails_closed(live_collection, field, replacement):
    client, manifest = live_collection
    payload = _payloads(client, manifest["collection_name"])[1]
    client.set_payload(
        collection_name=manifest["collection_name"],
        payload={field: replacement},
        points=[point_id_for_chunk(payload["chunk_id"])],
    )
    _assert_startup_rejects(client, manifest)


def test_missing_point_fails_closed(live_collection):
    client, manifest = live_collection
    payload = _payloads(client, manifest["collection_name"])[0]
    client.delete(
        collection_name=manifest["collection_name"],
        points_selector=PointIdsList(points=[point_id_for_chunk(payload["chunk_id"])]),
    )
    _assert_startup_rejects(client, manifest)


def test_duplicate_chunk_id_fails_closed(live_collection):
    client, manifest = live_collection
    first, second = _payloads(client, manifest["collection_name"])[:2]
    client.set_payload(
        collection_name=manifest["collection_name"],
        payload={"chunk_id": first["chunk_id"]},
        points=[point_id_for_chunk(second["chunk_id"])],
    )
    _assert_startup_rejects(client, manifest)


def test_conflicting_document_version_fails_closed(live_collection):
    client, manifest = live_collection
    payload = _payloads(client, manifest["collection_name"])[1]
    client.set_payload(
        collection_name=manifest["collection_name"],
        payload={"document_version": "ca:document-version:conflicting"},
        points=[point_id_for_chunk(payload["chunk_id"])],
    )
    _assert_startup_rejects(client, manifest)
