"""Unit test for extraction report serialization."""

from __future__ import annotations

import tempfile
from pathlib import Path

from data_agent_baseline.tools import doc_extract
from data_agent_baseline.tools.doc_extract import (
    ExtractedDoc,
    ExtractionReport,
    _load_cache,
    _save_cache,
)


def test_report_round_trip():
    doc = ExtractedDoc(
        stem="molecule",
        source_rel="doc/molecule.md",
        columns=["molecule_id", "element"],
        primary_key=["molecule_id"],
        rows=[{"molecule_id": "TR391", "element": "c"}],
        report=ExtractionReport(
            stem="molecule",
            row_count=1,
            key_scannable=True,
            missing_keys=["TR450"],
            empty_field_cells=0,
            field_fill_rate={"molecule_id": 1.0, "element": 1.0},
        ),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "molecule.json"
        _save_cache(path, doc, "hash")
        result = _load_cache(path, "hash")
        assert result is not None
        loaded, version = result
        assert version == doc_extract._EXTRACT_VERSION
        assert loaded.report is not None
        assert loaded.report.missing_keys == ["TR450"]
        assert loaded.report.key_scannable is True
        assert loaded.report.field_fill_rate["element"] == 1.0


if __name__ == "__main__":
    test_report_round_trip()
