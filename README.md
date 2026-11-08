# therapeutic-alliance
How to Build an NLP Pipeline in Python to Analyze Therapy Progress Notes

An educational Python project using **fictional English notes only**. Text features
and general-purpose classifiers do not measure therapeutic alliance, treatment
effectiveness, or a person's mental health.

## Quick start

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
therapy-notes sample --output data/synthetic_notes.csv
therapy-notes preprocess --input data/synthetic_notes.csv --output results/features.csv
pytest
```

The input CSV has `client_id`, `session` (positive integer), and `note` columns.
Each client/session pair must be unique. The sample command writes ten fictional
notes across two fictional clients. Preprocessing uses spaCy's English tokenizer
and sentence rules without downloading a language model. It preserves negation
and the original text and reports word counts, sentence lengths, lexical diversity,
and first-person word ratios.
