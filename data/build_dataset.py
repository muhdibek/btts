"""
build_dataset.py
----------------
Build the labelled BTTS training set from football-data.co.uk history.

    # download and build ten seasons of the big five leagues
    python -m data.build_dataset --leagues E0 SP1 D1 I1 F1 \
        --start-season 2015 --end-season 2024 \
        --out data/processed/btts_dataset.csv

    # rebuild from already-downloaded CSVs, no network
    python -m data.build_dataset --from-cache --out data/processed/btts_dataset.csv

    # list the division codes
    python -m data.build_dataset --list-leagues

Output: one row per played match, in kickoff order, with the pre-match
features from features.py and the `btts` label (1 = both teams scored).

Rows are chronological, so a train/test split must be by DATE, never random —
shuffling puts future matches in the training set and inflates every metric.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.football_data import (CACHE_DIR, LEAGUES, load_matches,      # noqa: E402
                                parse_season_csv, season_label)
from data.features import build_features, feature_columns              # noqa: E402


DEFAULT_LEAGUES = ["E0", "SP1", "D1", "I1", "F1", "N1"]
DEFAULT_OUT     = Path(__file__).resolve().parent / "processed" / "btts_dataset.csv"


def load_from_cache(cache_dir: Path) -> pd.DataFrame:
    """Parse every CSV already sitting in the cache — no network needed."""
    frames = []
    for path in sorted(Path(cache_dir).glob("*.csv")):
        # filenames look like E0_2425.csv
        try:
            code = path.stem.split("_")[1]
            start_year = 2000 + int(code[:2]) if int(code[:2]) < 90 else 1900 + int(code[:2])
        except (IndexError, ValueError):
            start_year = None
        frames.append(parse_season_csv(path, start_year=start_year))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("kickoff").reset_index(drop=True)


def summarise(df: pd.DataFrame) -> str:
    """Human-readable report on the built dataset."""
    if df.empty:
        return "Dataset is empty."

    features = feature_columns(df)
    missing  = df[features].isna().mean().sort_values(ascending=False)

    lines = [
        f"Matches:        {len(df):,}",
        f"Date range:     {df['kickoff'].min():%Y-%m-%d} → {df['kickoff'].max():%Y-%m-%d}",
        f"Divisions:      {', '.join(sorted(df['div'].dropna().unique()))}",
        f"Teams:          {pd.concat([df['home_team'], df['away_team']]).nunique():,}",
        f"BTTS base rate: {df['btts'].mean():.1%}",
        f"Features:       {len(features)}",
    ]
    if "mkt_p_over25" in df.columns:
        lines.append(f"Rows with odds: {df['mkt_p_over25'].notna().mean():.1%}")

    worst = missing.head(5)
    if len(worst) and worst.iloc[0] > 0:
        lines.append("Most-missing features: " +
                     ", ".join(f"{name} {rate:.1%}" for name, rate in worst.items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the labelled BTTS dataset.")
    parser.add_argument("--leagues", nargs="+", default=DEFAULT_LEAGUES,
                        help=f"division codes (default: {' '.join(DEFAULT_LEAGUES)})")
    parser.add_argument("--start-season", type=int, default=2015,
                        help="first season's starting year (default: 2015)")
    parser.add_argument("--end-season", type=int, default=2024,
                        help="last season's starting year, inclusive (default: 2024)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output CSV path")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR,
                        help="where raw season CSVs are cached")
    parser.add_argument("--from-cache", action="store_true",
                        help="build from cached CSVs only, without downloading")
    parser.add_argument("--force", action="store_true",
                        help="re-download seasons already cached")
    parser.add_argument("--min-history", type=int, default=0,
                        help="drop rows where either side has fewer than N prior matches")
    parser.add_argument("--list-leagues", action="store_true",
                        help="print the available division codes and exit")
    args = parser.parse_args(argv)

    if args.list_leagues:
        for code, name in LEAGUES.items():
            print(f"  {code:<4} {name}")
        return 0

    if args.from_cache:
        print(f"Loading cached CSVs from {args.cache_dir} …")
        matches = load_from_cache(args.cache_dir)
    else:
        seasons = range(args.start_season, args.end_season + 1)
        print(f"Fetching {len(args.leagues)} league(s) × "
              f"{args.end_season - args.start_season + 1} season(s) "
              f"({season_label(args.start_season)} → {season_label(args.end_season)}) …")
        matches = load_matches(args.leagues, seasons,
                               cache_dir=args.cache_dir, force=args.force)

    if matches.empty:
        print("No matches loaded. Check the league codes, the seasons, or "
              "network access to football-data.co.uk.", file=sys.stderr)
        return 1

    print(f"Loaded {len(matches):,} played matches. Building features …")
    dataset = build_features(matches)

    if args.min_history > 0:
        before  = len(dataset)
        dataset = dataset[
            (dataset["home_matches_played"] >= args.min_history) &
            (dataset["away_matches_played"] >= args.min_history)
        ].reset_index(drop=True)
        print(f"Dropped {before - len(dataset):,} rows with thin history "
              f"(< {args.min_history} prior matches).")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.out, index=False)

    print()
    print(summarise(dataset))
    print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
