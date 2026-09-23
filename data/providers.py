"""
providers.py
------------
One client, several football APIs.

Six providers were tried for this project and they differ only in trivia: a
base URL, where the key goes, what the endpoints are called, and what they name
their fields. None of that deserves its own module, and writing a separate
client per provider means writing the same request-and-parse logic again with a
fresh chance to get it wrong.

So a provider here is DATA, not code:

    Provider(base_url=..., auth="bearer", endpoints={...}, fixture_map={...})

Adding one is a dict, not a file. Correcting one after seeing a real response
is editing two lines of mapping.

    export FIVEDOLLAR_API_KEY=...
    python -m data.providers probe --provider fivedollar
    python -m data.providers fixtures --provider fivedollar --from 2026-10-09

WHY THE MAPPINGS ARE UNVERIFIED: every one of these hosts is unreachable from
the environment this was written in, so the field names follow each provider's
documentation. Parsing is deliberately tolerant — a name that has changed costs
one NaN column, never the run — and `probe` prints what actually came back plus
a field-by-field mapping check, so a mismatch is a two-line fix rather than a
debugging session.

Keys live in environment variables, never in this file and never in the repo.
Providers that carry the key in the query string make the URL itself a secret,
so it is redacted from every message this module prints.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


TIMEOUT_S = 30


@dataclass(frozen=True)
class Provider:
    """Everything that differs between one football API and another."""

    name:        str
    base_url:    str
    key_env:     str
    auth:        str                      # "bearer" | "header" | "query" | "none"
    endpoints:   dict[str, str]           # logical name → path or action
    fixture_map: dict[str, str] = field(default_factory=dict)
    odds_map:    dict[str, str] = field(default_factory=dict)
    auth_name:   str = "Authorization"    # header name, or query parameter name
    path_style:  str = "path"             # "path": /v1/fixtures · "action": ?action=x
    docs:        str = ""

    def key(self, explicit: str | None = None) -> str:
        value = explicit or os.environ.get(self.key_env, "")
        if not value and self.auth != "none":
            raise ProviderError(
                f"{self.name}: no API key. Set {self.key_env} in your environment "
                f"or pass --key. Never paste it into a shared chat."
            )
        return value


class ProviderError(RuntimeError):
    """The provider answered, but not with data."""


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

APIFOOTBALL = Provider(
    name="apifootball",
    base_url="https://apiv3.apifootball.com/",
    key_env="APIFOOTBALL_KEY",
    auth="query",
    auth_name="APIkey",
    path_style="action",
    endpoints={"countries": "get_countries", "leagues": "get_leagues",
               "fixtures": "get_events", "odds": "get_odds"},
    fixture_map={
        "match_id": "match_id", "match_date": "date", "match_time": "time",
        "match_hometeam_name": "home_team", "match_awayteam_name": "away_team",
        "match_hometeam_score": "fthg", "match_awayteam_score": "ftag",
        "league_name": "league", "country_name": "country",
    },
    odds_map={
        "match_id": "match_id", "odd_bookmakers": "bookmaker",
        "odd_1": "odds_home", "odd_x": "odds_draw", "odd_2": "odds_away",
        "o_over_25": "odds_over25", "o_under_25": "odds_under25",
        "bts_yes": "odds_btts_yes", "bts_no": "odds_btts_no",
    },
    docs="https://apiv3.apifootball.com/",
)

FIVEDOLLAR = Provider(
    name="fivedollar",
    base_url="https://api.5dollarfootballapi.com/v1/",
    key_env="FIVEDOLLAR_API_KEY",
    auth="bearer",
    path_style="path",
    endpoints={"leagues": "leagues", "fixtures": "fixtures", "odds": "odds"},
    # Guessed from the REST conventions this API follows; `probe` will say.
    fixture_map={
        "id": "match_id", "date": "date", "time": "time",
        "home_team": "home_team", "away_team": "away_team",
        "home_score": "fthg", "away_score": "ftag",
        "league": "league", "status": "status",
    },
    odds_map={
        "fixture_id": "match_id", "bookmaker": "bookmaker",
        "home": "odds_home", "draw": "odds_draw", "away": "odds_away",
        "over_2_5": "odds_over25", "under_2_5": "odds_under25",
        "btts_yes": "odds_btts_yes", "btts_no": "odds_btts_no",
    },
    docs="https://api.5dollarfootballapi.com/",
)

PROVIDERS: dict[str, Provider] = {p.name: p for p in (APIFOOTBALL, FIVEDOLLAR)}


def get_provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise ProviderError(f"unknown provider '{name}'; have: {', '.join(PROVIDERS)}")
    return PROVIDERS[name]


# ---------------------------------------------------------------------------
# Requesting
# ---------------------------------------------------------------------------

def _redact(text: str, key: str) -> str:
    return text.replace(key, f"<{'KEY'}>") if key and key in text else text


def request(provider: Provider, endpoint: str, key: str | None = None,
            **params: Any) -> Any:
    """
    Call one endpoint and return decoded JSON.

    Several of these APIs report errors with HTTP 200 and an error body, so the
    body is inspected rather than trusting the status code alone.
    """
    if endpoint not in provider.endpoints:
        raise ProviderError(f"{provider.name}: no endpoint '{endpoint}'")

    resolved = provider.endpoints[endpoint]
    token    = provider.key(key)
    headers  = {}
    query    = {k: v for k, v in params.items() if v is not None}

    if provider.path_style == "action":
        url = provider.base_url
        query["action"] = resolved
    else:
        url = provider.base_url.rstrip("/") + "/" + resolved.lstrip("/")

    if provider.auth == "bearer":
        headers[provider.auth_name] = f"Bearer {token}"
    elif provider.auth == "header":
        headers[provider.auth_name] = token
    elif provider.auth == "query":
        query[provider.auth_name] = token

    response = requests.get(url, params=query, headers=headers, timeout=TIMEOUT_S)
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError:
        raise ProviderError(
            f"{provider.name}/{endpoint}: expected JSON, got "
            f"{_redact(response.text[:200], token)}"
        ) from None

    if isinstance(payload, dict):
        if payload.get("error"):
            raise ProviderError(f"{provider.name}/{endpoint}: "
                                f"{payload.get('message', payload['error'])}")
        # REST providers usually wrap the list: {"data": [...]} / {"fixtures": [...]}
        for wrapper in ("data", "response", "result", "results", endpoint):
            if isinstance(payload.get(wrapper), list):
                return payload[wrapper]
    return payload


# ---------------------------------------------------------------------------
# Normalising
# ---------------------------------------------------------------------------

def _apply_map(raw: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Rename by mapping, filling anything absent with NaN.

    Tolerance is the point: these mappings are unverified, so a field that has
    been renamed upstream must cost one column, not the whole frame.
    """
    return pd.DataFrame({target: raw[source] if source in raw.columns else np.nan
                         for source, target in mapping.items()})


def normalise_fixtures(raw: pd.DataFrame, provider: Provider) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=list(provider.fixture_map.values()) + ["kickoff", "played"])

    out = _apply_map(raw, provider.fixture_map)
    stamp = (out.get("date", pd.Series(dtype=object)).astype(str) + " " +
             out.get("time", pd.Series(dtype=object)).fillna("00:00").astype(str))
    out["kickoff"] = pd.to_datetime(stamp, errors="coerce")

    for column in ("fthg", "ftag"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out["played"] = (out.get("fthg", pd.Series(np.nan, index=out.index)).notna() &
                     out.get("ftag", pd.Series(np.nan, index=out.index)).notna())
    return out.sort_values("kickoff").reset_index(drop=True)


def normalise_odds(raw: pd.DataFrame, provider: Provider) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=list(provider.odds_map.values()))

    out = _apply_map(raw, provider.odds_map)
    for column in out.columns:
        if column.startswith("odds_"):
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def best_prices(odds: pd.DataFrame) -> pd.DataFrame:
    """Best price per match across bookmakers — what a bettor would actually take."""
    if odds.empty or "match_id" not in odds.columns:
        return odds

    price_columns = [c for c in odds.columns if c.startswith("odds_")]
    if not price_columns:
        return odds
    return (odds.groupby("match_id")[price_columns].max().reset_index()
                .rename(columns={c: c.replace("odds_", "max_") for c in price_columns}))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def probe(provider: Provider, date_from: str, date_to: str,
          key: str | None = None, out: str | None = None) -> int:
    """Call every endpoint, print what came back, and check the mappings."""
    token = provider.key(key)
    print(f"Probing {provider.name} ({provider.base_url}) — "
          f"key from ${provider.key_env}, never printed.\n")

    report: dict[str, Any] = {}
    for endpoint in provider.endpoints:
        params = ({"from": date_from, "to": date_to}
                  if endpoint in ("fixtures", "odds") else {})
        try:
            payload = request(provider, endpoint, key=token, **params)
        except Exception as exc:                       # noqa: BLE001 — report, don't crash
            print(f"  {endpoint:<10} FAILED: {_redact(str(exc), token)}")
            report[endpoint] = {"error": _redact(str(exc), token)}
            continue

        rows  = payload if isinstance(payload, list) else [payload]
        first = rows[0] if rows else {}
        keys  = ", ".join(list(first)[:10]) if isinstance(first, dict) else type(first).__name__
        print(f"  {endpoint:<10} {len(rows):>5} rows · keys: {keys}")
        report[endpoint] = {"rows": len(rows), "sample": first}

    print("\nMapping check:")
    for label, endpoint, mapping in (("fixtures", "fixtures", provider.fixture_map),
                                     ("odds", "odds", provider.odds_map)):
        sample = report.get(endpoint, {}).get("sample")
        if not isinstance(sample, dict) or not sample:
            print(f"  {label:<9} no sample row returned")
            continue
        missing = [k for k in mapping if k not in sample]
        found   = len(mapping) - len(missing)
        print(f"  {label:<9} {found}/{len(mapping)} fields present"
              + (f" · missing: {', '.join(missing)}" if missing else " · all present"))
        if missing:
            print(f"            actual keys: {', '.join(list(sample)[:14])}")

    if out:
        Path(out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nFull sample written to {out} (no key inside) — "
              f"paste it back and the mapping can be corrected in two lines.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch from a football API provider.")
    parser.add_argument("command", choices=["probe", "fixtures", "odds", "list"])
    parser.add_argument("--provider", default="apifootball",
                        help=f"one of: {', '.join(PROVIDERS)}")
    parser.add_argument("--key", default=None, help="overrides the environment variable")
    parser.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD")
    parser.add_argument("--out", default=None, help="probe: write the full sample here")
    args = parser.parse_args(argv)

    if args.command == "list":
        for name, provider in PROVIDERS.items():
            configured = "set" if os.environ.get(provider.key_env) else "NOT SET"
            print(f"  {name:<12} {provider.base_url:<42} "
                  f"${provider.key_env} ({configured})")
        return 0

    today = pd.Timestamp.today()
    args.date_from = args.date_from or today.strftime("%Y-%m-%d")
    args.date_to   = args.date_to or (today + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    try:
        provider = get_provider(args.provider)
        if args.command == "probe":
            return probe(provider, args.date_from, args.date_to, args.key, args.out)

        endpoint = args.command
        raw = pd.DataFrame(request(provider, endpoint, key=args.key,
                                   **{"from": args.date_from, "to": args.date_to}))
        if endpoint == "fixtures":
            frame = normalise_fixtures(raw, provider)
            columns = [c for c in ("kickoff", "league", "home_team", "away_team",
                                   "fthg", "ftag") if c in frame.columns]
        else:
            frame = best_prices(normalise_odds(raw, provider))
            columns = list(frame.columns)

        if frame.empty:
            print(f"No {endpoint} returned for that window.")
            return 1
        print(f"{len(frame):,} rows")
        print(frame[columns].head(20).to_string(index=False))
        return 0
    except ProviderError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
