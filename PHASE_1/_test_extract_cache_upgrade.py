"""Unit tests for:
A. extract cache version reuse + deterministic upgrade (no LLM re-extraction)
B. serial extraction path
C. OpenAI client timeout / max_retries bounds
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

from data_agent_baseline.agents.model import ModelMessage, OpenAIModelAdapter
from data_agent_baseline.tools import doc_extract
from data_agent_baseline.tools.doc_extract import (
    ExtractedDoc,
    _accepted_source_hashes,
    _hash_material,
    _load_cache,
    _save_cache,
    _source_hash,
    _source_hash_from_material,
    extract_document,
)

_PARA_ALIAS = (
    "The general Business program is the most popular choice among freshmen "
    "at this school and it enrolls many students every year."
)
_PARA_REGISTRY = (
    "Business (Registry ID: R1) is a program for Business students, offering "
    "courses in accounting, finance, marketing, and management."
)


def _make_context(tmpdir: str) -> tuple[Path, Path]:
    context_dir = Path(tmpdir) / "context"
    (context_dir / "doc").mkdir(parents=True)
    (context_dir / "knowledge.md").write_text("Registry of programs.\n", encoding="utf-8")
    doc_path = context_dir / "doc" / "major.md"
    doc_path.write_text(f"{_PARA_ALIAS}\n\n{_PARA_REGISTRY}\n", encoding="utf-8")
    return context_dir, doc_path


def _v3_cache_payload(context_dir: Path, doc_path: Path) -> tuple[Path, str]:
    doc_bytes, knowledge_bytes = _hash_material(doc_path, context_dir)
    v3_hash = _source_hash_from_material(doc_bytes, knowledge_bytes, 3)
    doc = ExtractedDoc(
        stem="major",
        source_rel="doc/major.md",
        columns=["registry_id", "program_name"],
        primary_key=["registry_id"],
        rows=[{"registry_id": "R1", "program_name": "general Business"}],
    )
    cache_dir = doc_extract.extract_cache_dir(context_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "doc_major_md.json"
    cache_file.write_text(
        json.dumps(
            {
                "version": 3,
                "source_hash": v3_hash,
                "stem": doc.stem,
                "source_rel": doc.source_rel,
                "columns": doc.columns,
                "primary_key": doc.primary_key,
                "rows": doc.rows,
                "report": None,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return cache_file, v3_hash


def test_v3_cache_accepted_and_upgraded_without_model():
    """A: v3 cache loads, Registry fix applied deterministically, re-saved as v4."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_dir, doc_path = _make_context(tmpdir)
        cache_file, _ = _v3_cache_payload(context_dir, doc_path)

        # model=None: any LLM call would fail loudly; none may happen.
        doc = extract_document(doc_path, context_dir, None)
        assert doc is not None
        assert doc.rows[0]["program_name"] == "Business"

        # Re-saved at the current version with the current hash.
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        assert payload["version"] == doc_extract._EXTRACT_VERSION
        assert payload["source_hash"] == _source_hash(doc_path, context_dir)
        assert payload["rows"][0]["program_name"] == "Business"


def test_accepted_hashes_cover_supported_versions():
    with tempfile.TemporaryDirectory() as tmpdir:
        context_dir, doc_path = _make_context(tmpdir)
        accepted = _accepted_source_hashes(doc_path, context_dir)
        doc_bytes, knowledge_bytes = _hash_material(doc_path, context_dir)
        for version in range(doc_extract._MIN_CACHE_VERSION, doc_extract._EXTRACT_VERSION + 1):
            assert _source_hash_from_material(doc_bytes, knowledge_bytes, version) in accepted


def test_stale_source_hash_rejected():
    """Cache whose source content changed must miss even with a known version."""
    with tempfile.TemporaryDirectory() as tmpdir:
        context_dir, doc_path = _make_context(tmpdir)
        cache_file, _ = _v3_cache_payload(context_dir, doc_path)
        doc_path.write_text(
            f"{_PARA_ALIAS}\n\n{_PARA_REGISTRY}\n\nExtra paragraph changes the hash now.\n",
            encoding="utf-8",
        )
        assert _load_cache(cache_file, _accepted_source_hashes(doc_path, context_dir)) is None
        # model=None + cache miss -> no extraction possible.
        assert extract_document(doc_path, context_dir, None) is None


def test_too_old_cache_version_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        context_dir, doc_path = _make_context(tmpdir)
        cache_file, v3_hash = _v3_cache_payload(context_dir, doc_path)
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        payload["version"] = doc_extract._MIN_CACHE_VERSION - 1
        doc_bytes, knowledge_bytes = _hash_material(doc_path, context_dir)
        payload["source_hash"] = _source_hash_from_material(
            doc_bytes, knowledge_bytes, doc_extract._MIN_CACHE_VERSION - 1
        )
        cache_file.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        assert _load_cache(cache_file, _accepted_source_hashes(doc_path, context_dir)) is None
        assert v3_hash != payload["source_hash"]


def test_serial_extraction_skips_thread_pool():
    """B: EXTRACT_CONCURRENCY=1 must not spin up a ThreadPoolExecutor."""
    schema_response = json.dumps(
        {
            "columns": [{"name": "registry_id"}, {"name": "program_name"}],
            "primary_key": ["registry_id"],
        }
    )
    row_response = json.dumps(
        {"skip": False, "values": {"registry_id": "R1", "program_name": "Business"}}
    )
    skip_response = json.dumps({"skip": True, "values": {}})

    with tempfile.TemporaryDirectory() as tmpdir:
        context_dir, doc_path = _make_context(tmpdir)
        old_concurrency = doc_extract.EXTRACT_CONCURRENCY
        old_delay = doc_extract.EXTRACT_CALL_DELAY_SECONDS
        doc_extract.EXTRACT_CONCURRENCY = 1
        doc_extract.EXTRACT_CALL_DELAY_SECONDS = 0.0
        try:
            with mock.patch.object(doc_extract, "ThreadPoolExecutor") as pool_mock:
                from data_agent_baseline.agents.model import ScriptedModelAdapter

                # schema + 2 paragraphs + (maybe) recovery-pass calls.
                model = ScriptedModelAdapter(
                    [schema_response, row_response, skip_response, skip_response, skip_response]
                )
                doc = extract_document(doc_path, context_dir, model)
                assert doc is not None
                assert doc.rows[0]["registry_id"] == "R1"
                pool_mock.assert_not_called()
        finally:
            doc_extract.EXTRACT_CONCURRENCY = old_concurrency
            doc_extract.EXTRACT_CALL_DELAY_SECONDS = old_delay


class _FakeMessage:
    content = '{"thought": "t", "action": "answer", "action_input": {}}'


class _FakeChoice:
    message = _FakeMessage()


class _FakeResponse:
    choices = [_FakeChoice()]


def test_client_timeout_and_retries_bounds():
    """C: OpenAI client is constructed with hard timeout and retry bounds."""
    with mock.patch("data_agent_baseline.agents.model.OpenAI") as mock_openai:
        mock_openai.return_value.chat.completions.create.return_value = _FakeResponse()
        adapter = OpenAIModelAdapter(
            model="m", api_base="http://localhost", api_key="k", temperature=0.0
        )
        assert adapter.request_timeout == 60.0
        assert adapter.max_retries == 1
        adapter.complete([ModelMessage(role="user", content="hi")])
        _, kwargs = mock_openai.call_args
        assert kwargs["timeout"] == 60.0
        assert kwargs["max_retries"] == 1


def test_client_timeout_and_retries_overridable():
    adapter = OpenAIModelAdapter(
        model="m",
        api_base="http://localhost",
        api_key="k",
        temperature=0.0,
        request_timeout=15,
        max_retries=0,
    )
    assert adapter.request_timeout == 15.0
    assert adapter.max_retries == 0


if __name__ == "__main__":
    test_v3_cache_accepted_and_upgraded_without_model()
    test_accepted_hashes_cover_supported_versions()
    test_stale_source_hash_rejected()
    test_too_old_cache_version_rejected()
    test_serial_extraction_skips_thread_pool()
    test_client_timeout_and_retries_bounds()
    test_client_timeout_and_retries_overridable()
    print("all cache-upgrade / serial / client-bounds tests passed")
