"""Join model scores to linguistic features without inferring clinical outcomes."""

import math

import pandas as pd

from .models import EMOTIONS
from .preprocessing import preprocess


def checked_predictions(classifier, texts, expected):
    predictions = classifier.predict(texts)
    if len(predictions) != len(texts):
        raise ValueError("Model returned the wrong number of predictions.")
    for prediction in predictions:
        scores = prediction["scores"]
        if set(scores) != set(expected):
            raise ValueError(f"Unexpected model labels: {sorted(scores)}")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
            raise ValueError("Model probabilities must be finite and between zero and one.")
        if not math.isclose(sum(scores.values()), 1, abs_tol=1e-4):
            raise ValueError("Model probabilities must sum to one.")
    return predictions


def analyze(frame, sentiment, emotion):
    result = preprocess(frame)
    texts = result["note"].tolist()
    sentiments = checked_predictions(sentiment, texts, ("positive", "negative"))
    emotions = checked_predictions(emotion, texts, EMOTIONS)
    rows = []
    for sent, emo in zip(sentiments, emotions):
        rows.append({
            "sentiment_label": max(sent["scores"], key=sent["scores"].get),
            "positive_score": sent["scores"]["positive"],
            "negative_score": sent["scores"]["negative"],
            "sentiment_chunks": sent["chunks"],
            "emotion_label": max(emo["scores"], key=emo["scores"].get),
            "emotion_chunks": emo["chunks"],
            **{f"emotion_{label}": emo["scores"][label] for label in EMOTIONS},
        })
    return pd.concat([result, pd.DataFrame(rows)], axis=1)
