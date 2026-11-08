import os

import pytest

from therapeutic_alliance.analysis import analyze
from therapeutic_alliance.data import synthetic_notes
from therapeutic_alliance.models import (
    EMOTION_MODEL,
    EMOTIONS,
    SENTIMENT_MODEL,
    TransformerClassifier,
)


class FixtureClassifier:
    """Deterministic contract fixture; never used for the CLI or demo."""
    def __init__(self, scores):
        self.metadata = {"model": "test-fixture", "revision": "test"}
        self.scores = scores
        self.seen = []

    def predict(self, texts):
        self.seen = texts
        return [{"scores": self.scores, "chunks": 1} for _ in texts]


def fixtures():
    return (FixtureClassifier({"positive": .75, "negative": .25}),
            FixtureClassifier({label: 1 / 7 for label in EMOTIONS}))


def test_analysis_retains_text_and_all_probabilities():
    sentiment, emotion = fixtures()
    frame = synthetic_notes().iloc[::-1]
    result = analyze(frame, sentiment, emotion)
    assert result.client_id.tolist() == synthetic_notes().client_id.tolist()
    assert sentiment.seen == emotion.seen == synthetic_notes().note.tolist()
    assert result.positive_score.tolist() == [.75] * 10
    assert result.sentiment_label.tolist() == ["positive"] * 10
    assert result.filter(regex="^emotion_(anger|disgust|fear|joy|neutral|sadness|surprise)$").shape == (10, 7)


@pytest.mark.parametrize("scores", [
    {"positive": .2}, {"positive": float("nan"), "negative": .2},
    {"positive": .8, "negative": .8}, {"positive": -1, "negative": 2},
])
def test_rejects_invalid_model_output(scores):
    _, emotion = fixtures()
    with pytest.raises(ValueError):
        analyze(synthetic_notes(), FixtureClassifier(scores), emotion)


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="Set RUN_MODEL_TESTS=1")
def test_real_models_and_long_note():
    import torch
    torch.set_num_threads(2)
    sentiment = TransformerClassifier(SENTIMENT_MODEL)
    emotion = TransformerClassifier(EMOTION_MODEL)
    frame = synthetic_notes().iloc[:2].copy()
    frame.loc[frame.index[1], "note"] = "I am not happy about this difficult week. " * 90
    result = analyze(frame, sentiment, emotion)
    assert result.iloc[1].sentiment_chunks > 1
    assert result.iloc[1].emotion_chunks > 1
    assert result.positive_score.between(0, 1).all()
    assert sentiment.metadata["revision"]
    assert emotion.metadata["revision"]
