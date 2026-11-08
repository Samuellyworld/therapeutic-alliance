"""Command-line entry point."""

import argparse
from pathlib import Path

from .data import load_notes, synthetic_notes
from .preprocessing import preprocess


def main(argv=None):
    parser = argparse.ArgumentParser(description="Explore fictional therapy-note language.")
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="Write ten fictional sample notes.")
    sample.add_argument("--output", type=Path, default=Path("data/synthetic_notes.csv"))
    prep = commands.add_parser("preprocess", help="Extract spaCy linguistic features offline.")
    prep.add_argument("--input", type=Path, required=True)
    prep.add_argument("--output", type=Path, default=Path("results/features.csv"))
    for name, help_text in (("analyze", "Analyze a fictional notes CSV with real models."),
                            ("demo", "Run the complete pipeline on ten fictional notes.")):
        command = commands.add_parser(name, help=help_text)
        if name == "analyze":
            command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, default=Path("results") / name)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--local-files-only", action="store_true", help="Use cached model files only.")
    args = parser.parse_args(argv)
    try:
        if args.command in ("analyze", "demo"):
            if args.batch_size < 1:
                raise ValueError("--batch-size must be positive.")
            from .analysis import analyze
            from .models import EMOTION_MODEL, SENTIMENT_MODEL, TransformerClassifier
            from .reporting import LIMITATION, write_report

            # Validate before downloading or loading weights.
            source = synthetic_notes() if args.command == "demo" else load_notes(args.input)
            preprocess(source)
            print(LIMITATION)
            print("Loading sentiment and emotion models on CPU…", flush=True)
            options = {"batch_size": args.batch_size, "local_files_only": args.local_files_only}
            sentiment = TransformerClassifier(SENTIMENT_MODEL, **options)
            emotion = TransformerClassifier(EMOTION_MODEL, **options)
            result = analyze(source, sentiment, emotion)
            report = write_report(result, args.output, {
                "sentiment": sentiment.metadata, "emotion": emotion.metadata,
            })
            print(f"Analyzed {len(result)} notes. Open {report.resolve()}")
            return
        frame = synthetic_notes() if args.command == "sample" else preprocess(load_notes(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Wrote {len(frame)} fictional notes to {args.output}")


if __name__ == "__main__":
    main()
