"""
apifootball.py
--------------
Client for APIFootball v3 (https://apiv3.apifootball.com).

This is the only source tried in this project that carries **bookmaker odds**
for matches not yet played. Everything else here — openfootball, the
football-charts connector, the GitHub archives — gives fixtures, results and
model probabilities, but no forward prices. Prices are what turn "the model
says 62%" into "the model says 62% and the market says 55%", which is the
difference between a description and a claim.

    export APIFOOTBALL_KEY=your_key
    python -m data.apifootball probe            # what does the API actually return?
    python -m data.apifootball odds --from 2026-10-09 --to 2026-10-12

UNVERIFIED AGAINST THE LIVE API. It was written where the host is unreachable,
so the request shapes follow the documented API and the parsing is deliberately
tolerant: unknown keys are ignored, missing ones become NaN rather than raising,
and `probe` prints the raw payload so any mismatch is visible in one run rather
than debugged through a stack trace. Run `probe` first and the output will say
whether the assumptions below hold.

Key handling: read from APIFOOTBALL_KEY, never committed, never logged. The
API takes it as a query parameter, so the URL itself is a secret — redacted
in every message this module prints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


BASE_URL  = "https://apiv3.apifootball.com/"
TIMEOUT_S = 30
KEY_ENV   = "APIFOOTBALL_KEY"


class ApiFootballError(RuntimeError):
    """The API answered, but not with data — a bad key, a quota, or an error body."""


def _redact(text: str, key: str | None) -> str:
    """Never let the key reach a log line or a traceback."""
    if key and key in text:
        return text.replace(key, "<APIFOOTBALL_KEY>")
    return text


def api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get(KEY_ENV, "")
    if not key:
        raise ApiFootballError(
            f"No API key. Set {KEY_ENV} in your environment, or pass --key. "
            "Do not paste it into a shared chat — it travels in the URL."
        )
    return key


def call(action: str, key: str | None = None, **params: Any) -> Any:
    """
    One API call. Returns the decoded JSON.

    The API answers errors with HTTP 200 and a body like
    {"error": 404, "message": "..."}, so the body is checked, not just the
    status code.
    """
    key = api_key(key)
    query = {"action": action, "APIkey": key, **{k: v for k, v in params.items()
                                                 if v is not None}}
    response = requests.get(BASE_URL, params=query, timeout=TIMEOUT_S)
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError:
        raise ApiFootballError(
            f"{action}: expected JSON, got {_redact(response.text[:200], key)}"
        ) from None

    if isinstance(payload, dict) and payload.get("error"):
        raise ApiFootballError(f"{action}: {payload.get('message', payload['error'])}")
    return payload


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def get_countries(key: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(call("get_countries", key=key))


def get_leagues(country_id: str | int | None = None, key: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(call("get_leagues", key=key, country_id=country_id))


def get_events(date_from: str, date_to: str, league_id: str | int | None = None,
               key: str | None = None) -> pd.DataFrame:
    """Fixtures and results in a date window (YYYY-MM-DD)."""
    return pd.DataFrame(call("get_events", key=key, **{"from": date_from, "to": date_to,
                                                       "league_id": league_id}))


def get_odds(date_from: str, date_to: str, match_id: str | int | None = None,
             key: str | None = None) -> pd.DataFrame:
    """
    Pre-match odds. One row per bookmaker per match in the documented shape,
    with 1X2 in odd_1 / odd_x / odd_2 and over/under and BTTS alongside.
    """
    return pd.DataFrame(call("get_odds", key=key, **{"from": date_from, "to": date_to,
                                                     "match_id": match_id}))


# ---------------------------------------------------------------------------
# Normalising
# ---------------------------------------------------------------------------

FIXTURE_COLUMNS = {
    "match_id": "match_id", "match_date": "date", "match_time": "time",
    "match_hometeam_name": "home_team", "match_awayteam_name": "away_team",
    "match_hometeam_score": "fthg", "match_awayteam_score": "ftag",
    "league_name": "league", "country_name": "country", "match_status": "status",
}


def normalise_fixtures(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Map an events payload into this repo's schema.

    Columns the payload does not carry are filled with NaN rather than raising,
    so a changed field name degrades one column instead of the whole run.
    """
    if raw.empty:
        return pd.DataFrame(columns=list(FIXTURE_COLUMNS.values()) + ["kickoff", "played"])

    out = pd.DataFrame({target: raw[source] if source in raw.columns else np.nan
                        for source, target in FIXTURE_COLUMNS.items()})

    out["kickoff"] = pd.to_datetime(
        out["date"].astype(str) + " " + out["time"].fillna("00:00").astype(str),
        errors="coerce")
    for column in ("fthg", "ftag"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out["played"] = out["fthg"].notna() & out["ftag"].notna()

    return out.sort_values("kickoff").reset_index(drop=True)


ODDS_COLUMNS = {
    "match_id": "match_id", "odd_bookmakers": "bookmaker",
    "odd_1": "odds_home", "odd_x": "odds_draw", "odd_2": "odds_away",
    "o_over_25": "odds_over25", "o_under_25": "odds_under25",
    "bts_yes": "odds_btts_yes", "bts_no": "odds_btts_no",
}


def normalise_odds(raw: pd.DataFrame) -> pd.DataFrame:
    """Map an odds payload into the schema models/market_test.py already reads."""
    if raw.empty:
        return pd.DataFrame(columns=list(ODDS_COLUMNS.values()))

    out = pd.DataFrame({target: raw[source] if source in raw.columns else np.nan
                        for source, target in ODDS_COLUMNS.items()})
    for column in out.columns:
        if column not in ("match_id", "bookmaker"):
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def best_prices(odds: pd.DataFrame) -> pd.DataFrame:
    """
    The best price per match across bookmakers — what a bettor would actually
    take, and the harder benchmark for any model.
    """
    if odds.empty:
        return odds

    price_columns = [c for c in odds.columns if c.startswith("odds_")]
    return (odds.groupby("match_id")[price_columns].max().reset_index()
                .rename(columns={c: c.replace("odds_", "max_") for c in price_columns}))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _probe(args) -> int:
    """Show what the API actually returns, so the mapping above can be checked."""
    key = api_key(args.key)
    print("Probing APIFootball v3 — key read from the environment, never printed.\n")

    checks = [
        ("get_countries", {}),
        ("get_leagues",   {}),
        ("get_events",    {"from": args.date_from, "to": args.date_to}),
        ("get_odds",      {"from": args.date_from, "to": args.date_to}),
    ]

    report = {}
    for action, params in checks:
        try:
            payload = call(action, key=key, **params)
        except Exception as exc:                       # noqa: BLE001 — report, don't crash
            print(f"  {action:<14} FAILED: {_redact(str(exc), key)}")
            report[action] = {"error": _redact(str(exc), key)}
            continue

        rows = payload if isinstance(payload, list) else [payload]
        first = rows[0] if rows else {}
        print(f"  {action:<14} {len(rows):>5} rows · keys: "
              f"{', '.join(list(first)[:10]) if isinstance(first, dict) else type(first).__name__}")
        report[action] = {"rows": len(rows), "sample": first}

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nFull sample written to {args.out} — paste it back if a mapping looks wrong.")

    events = report.get("get_events", {}).get("sample") or {}
    odds   = report.get("get_odds", {}).get("sample") or {}
    print("\nMapping check:")
    for label, sample, mapping in (("fixtures", events, FIXTURE_COLUMNS),
                                   ("odds", odds, ODDS_COLUMNS)):
        if not isinstance(sample, dict) or not sample:
            print(f"  {label}: no sample row returned")
            continue
        missing = [k for k in mapping if k not in sample]
        print(f"  {label}: {len(mapping) - len(missing)}/{len(mapping)} expected fields present"
              + (f" · missing {', '.join(missing)}" if missing else " · all present"))
    return 0


def _odds(args) -> int:
    raw = get_odds(args.date_from, args.date_to, key=args.key)
    odds = normalise_odds(raw)
    if odds.empty:
        print("No odds returned for that window.")
        return 1
    print(f"{len(odds):,} odds rows · {odds['match_id'].nunique()} matches · "
          f"{odds['bookmaker'].nunique()} bookmakers")
    print(best_prices(odds).head(20).to_string(index=False))
    return 0


def _fixtures(args) -> int:
    fixtures = normalise_fixtures(get_events(args.date_from, args.date_to, key=args.key))
    if fixtures.empty:
        print("No fixtures returned for that window.")
        return 1
    print(f"{len(fixtures):,} fixtures · {int(fixtures['played'].sum())} played")
    print(fixtures[["kickoff", "league", "home_team", "away_team", "fthg", "ftag"]]
          .head(20).to_string(index=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="APIFootball v3 client.")
    parser.add_argument("--key", default=None,
                        help=f"API key (default: ${KEY_ENV})")
    parser.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD")
    parser.add_argument("--out", default=None, help="probe: write the full sample here")
    parser.add_argument("command", choices=["probe", "odds", "fixtures"])
    args = parser.parse_args(argv)

    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    args.date_from = args.date_from or today
    args.date_to   = args.date_to or (pd.Timestamp.today() + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        return {"probe": _probe, "odds": _odds, "fixtures": _fixtures}[args.command](args)
    except ApiFootballError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
