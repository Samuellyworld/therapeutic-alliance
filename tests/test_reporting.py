import json

import matplotlib.pyplot as plt
import pandas as pd
from test_analysis import fixtures

from therapeutic_alliance.analysis import analyze
from therapeutic_alliance.cli import main
from therapeutic_alliance.data import synthetic_notes
from therapeutic_alliance.models import SENTIMENT_MODEL
from therapeutic_alliance.reporting import write_report


def test_report_exports_and_escapes_ids(tmp_path):
    source = synthetic_notes()
    source["client_id"] = source["client_id"].replace("fictional-A", "<script>alert(1)</script>")
    sentiment, emotion = fixtures()
    result = analyze(source, sentiment, emotion)
    report = write_report(result, tmp_path, {"sentiment": sentiment.metadata, "emotion": emotion.metadata})
    page = report.read_text()
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page
    assert "data:image/png;base64," in page
    assert "not clinical" in page
    assert (tmp_path / "overview.png").read_bytes().startswith(b"\x89PNG")
    assert len(pd.read_csv(tmp_path / "analysis.csv")) == 10
    assert pd.read_csv(tmp_path / "client_summary.csv").sessions.tolist() == [5, 5]
    metadata = json.loads((tmp_path / "manifest.json").read_text())
    assert metadata["note_count"] == 10
    assert len(metadata["input_sha256"]) == 64
    assert not plt.get_fignums()


def test_demo_cli_writes_report(monkeypatch, tmp_path):
    sentiment, emotion = fixtures()
    monkeypatch.setattr("therapeutic_alliance.models.TransformerClassifier",
                        lambda model_id, **kwargs: sentiment if model_id == SENTIMENT_MODEL else emotion)
    main(["demo", "--output", str(tmp_path)])
    assert (tmp_path / "report.html").exists()


def test_analyze_cli_validates_before_loading_models(tmp_path, capsys):
    import pytest
    path = tmp_path / "bad.csv"
    path.write_text("client_id,session,note\nfictional-A,1,...\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["analyze", "--input", str(path)])
    assert error.value.code == 2
    assert "alphabetic" in capsys.readouterr().err
