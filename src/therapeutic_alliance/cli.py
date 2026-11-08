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
    args = parser.parse_args(argv)
    try:
        frame = synthetic_notes() if args.command == "sample" else preprocess(load_notes(args.input))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Wrote {len(frame)} fictional notes to {args.output}")


if __name__ == "__main__":
    main()
