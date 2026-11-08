"""Portable CSV, PNG, HTML, and provenance outputs."""

import base64
import hashlib
import html
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .models import EMOTIONS

LIMITATION = (
    "Educational text exploration only. General-purpose model scores are not clinical "
    "measurements, diagnoses, evidence of therapeutic alliance, or treatment outcomes. "
    "Use fictional English notes only."
)


def plot_overview(frame, path):
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(3, 1, figsize=(11, 13), constrained_layout=True)
        fig.patch.set_facecolor("#f5f8fc")
        clients = list(frame.groupby("client_id", sort=True))
        for client, notes in clients:
            notes = notes.sort_values("session")
            axes[0].plot(notes.session, notes.positive_score, "o-", label=client, linewidth=2)
            axes[2].plot(notes.session, notes.mean_sentence_length, "o-", label=client, linewidth=2)
        axes[0].set(title="Sentiment across fictional sessions", ylabel="Positive model score", ylim=(0, 1))
        axes[0].legend(frameon=False)
        for ax in (axes[0], axes[2]):
            ax.set_xlabel("Session (within each client)")
            ax.set_xticks(sorted(frame.session.unique()))
            ax.grid(alpha=.18)
        emotion_means = frame.groupby("client_id")[[f"emotion_{e}" for e in EMOTIONS]].mean()
        image = axes[1].imshow(emotion_means.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        axes[1].set(title="Mean emotion model scores · descriptive averages")
        axes[1].set_xticks(range(len(EMOTIONS)), EMOTIONS)
        axes[1].set_yticks(range(len(emotion_means)), emotion_means.index)
        for row in range(len(emotion_means)):
            for col in range(len(EMOTIONS)):
                value = emotion_means.iloc[row, col]
                axes[1].text(col, row, f"{value:.2f}", ha="center", va="center",
                             color="white" if value > .55 else "#12253f")
        fig.colorbar(image, ax=axes[1], shrink=.8, label="Model score")
        axes[2].set(title="Language patterns across fictional sessions", ylabel="Words per sentence")
        fig.suptitle("Therapeutic Alliance · fictional text explorer", fontsize=19, fontweight="bold")
        fig.supxlabel("Text scores describe wording; they do not measure therapy progress.", fontsize=10)
        try:
            fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
        finally:
            plt.close(fig)


def write_report(frame, output: Path, model_metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "analysis.csv", index=False)
    summary = frame.groupby("client_id", sort=True).agg(
        sessions=("session", "count"),
        mean_positive_score=("positive_score", "mean"),
        mean_word_count=("word_count", "mean"),
        mean_lexical_diversity=("lexical_diversity", "mean"),
    ).reset_index()
    summary.to_csv(output / "client_summary.csv", index=False)
    plot_overview(frame, output / "overview.png")
    records = frame[["client_id", "session", "note"]].to_json(orient="records", force_ascii=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "note_count": len(frame),
        "input_sha256": hashlib.sha256(records.encode("utf-8")).hexdigest(),
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("therapeutic-alliance", "spacy", "pandas", "matplotlib")},
        "models": model_metadata,
        "limitation": LIMITATION,
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    png = base64.b64encode((output / "overview.png").read_bytes()).decode("ascii")
    columns = ["client_id", "session", "sentiment_label", "positive_score", "emotion_label", "word_count"]
    table = frame[columns].to_html(index=False, escape=True, float_format=lambda value: f"{value:.3f}")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Therapeutic Alliance | Fictional note report</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf3fa;color:#172c46;font:16px/1.65 system-ui,sans-serif}}
main{{max-width:1100px;margin:48px auto;padding:0 24px}}header{{background:#12253f;color:#fff;padding:40px;border-radius:20px}}
.eyebrow{{color:#62dac9;letter-spacing:.16em;font-size:12px;font-weight:700}}h1{{font-size:clamp(28px,5vw,44px);line-height:1.15}}
.notice{{border-left:4px solid #19ab96;background:white;padding:20px 24px;margin:24px 0;border-radius:4px 12px 12px 4px}}
section{{background:#fff;border-radius:16px;padding:24px;margin-top:24px}}img{{width:100%;height:auto}}
.table{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px;text-align:left;border-bottom:1px solid #e1e8f0}}
th{{background:#edf3fa}}footer{{font-size:13px;color:#52667c;margin:24px 0}}
</style></head><body><main>
<header><div class="eyebrow">FICTIONAL DATA · LOCAL NLP</div><h1>Language patterns,<br>session by session.</h1>
<p>{len(frame)} notes · {frame.client_id.nunique()} fictional clients · sentiment, emotion, and linguistic features</p></header>
<p class="notice">{html.escape(LIMITATION)}</p>
<section><h2>Explore the wording</h2><img alt="Sentiment scores, emotion averages, and sentence lengths for each fictional client" src="data:image/png;base64,{png}"></section>
<section><h2>Session results</h2><p>Scores are model outputs, not calibrated probabilities of a person's emotional state.</p><div class="table">{table}</div></section>
<footer>Full precision results: analysis.csv · Descriptive client averages: client_summary.csv · Model revisions and run details: manifest.json</footer>
</main></body></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
    return output / "report.html"
