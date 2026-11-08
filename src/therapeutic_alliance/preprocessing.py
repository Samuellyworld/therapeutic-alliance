"""Lightweight spaCy tokenization; original text stays intact for classifiers."""

from functools import lru_cache

import pandas as pd
import spacy

from .data import validate_notes

FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"}


@lru_cache(maxsize=1)
def language():
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    result = validate_notes(frame)
    features = []
    for doc in language().pipe(result["note"].tolist()):
        words = [token for token in doc if token.is_alpha]
        count = len(words)
        if not count:
            raise ValueError("Each note must contain at least one alphabetic word.")
        sentences = sum(1 for sent in doc.sents if any(t.is_alpha for t in sent))
        features.append({
            "word_count": count,
            "sentence_count": sentences,
            "mean_sentence_length": count / sentences,
            "lexical_diversity": len({t.lower_ for t in words}) / count,
            "first_person_ratio": sum(t.lower_ in FIRST_PERSON for t in words) / count,
        })
    return pd.concat([result, pd.DataFrame(features)], axis=1)
