import pandas as pd
import pytest

from therapeutic_alliance.cli import main
from therapeutic_alliance.data import load_notes, synthetic_notes, validate_notes
from therapeutic_alliance.preprocessing import preprocess


def test_fictional_sample_is_reproducible():
    pd.testing.assert_frame_equal(synthetic_notes(), synthetic_notes())
    assert len(synthetic_notes()) == 10
    assert synthetic_notes().groupby("client_id").size().tolist() == [5, 5]


@pytest.mark.parametrize("change", [
    {"session": 0}, {"session": 1.5}, {"session": float("inf")},
    {"note": " "}, {"note": None}, {"note": 23}, {"client_id": ""},
])
def test_rejects_invalid_input(change):
    row = {"client_id": "fictional", "session": 1, "note": "I feel okay."}
    row.update(change)
    with pytest.raises(ValueError):
        validate_notes(pd.DataFrame([row]))


def test_missing_empty_and_duplicate_rows():
    for frame in [pd.DataFrame(), synthetic_notes().iloc[:0],
                  pd.concat([synthetic_notes(), synthetic_notes()])]:
        with pytest.raises(ValueError):
            validate_notes(frame)


def test_features_preserve_negation_and_text():
    text = "I am not happy. I am worried!"
    frame = preprocess(pd.DataFrame([{"client_id": "001", "session": 1, "note": text}]))
    row = frame.iloc[0]
    assert row.note == text
    assert row.word_count == 7
    assert row.sentence_count == 2
    assert row.mean_sentence_length == 3.5
    assert row.first_person_ratio == 2 / 7
    assert row.lexical_diversity == 5 / 7


def test_rejects_wordless_note():
    frame = synthetic_notes().iloc[:1].copy()
    frame["note"] = "... 123"
    with pytest.raises(ValueError, match="alphabetic"):
        preprocess(frame)


def test_csv_and_cli_roundtrip(tmp_path):
    sample, output = tmp_path / "sample.csv", tmp_path / "nested" / "features.csv"
    main(["sample", "--output", str(sample)])
    main(["preprocess", "--input", str(sample), "--output", str(output)])
    assert len(pd.read_csv(output)) == 10
    sample.write_text('client_id,session,note\n001,1,NA\n', encoding="utf-8")
    assert load_notes(sample).iloc[0].client_id == "001"
    assert load_notes(sample).iloc[0].note == "NA"


def test_cli_error_is_actionable(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["preprocess", "--input", str(tmp_path / "missing.csv")])
    assert error.value.code == 2
    assert "Error:" in capsys.readouterr().err
