# ⚽ Match Outlook

A local Streamlit dashboard showing today's football fixtures, each side's recent
form from real results, and a statistical model's probability for the match.

**It is an information tool, not a betting tool** — and that is a conclusion, not
a disclaimer. It was built as a betting tool. The models were then tested properly,
against closing bookmaker odds on 17,697 out-of-sample matches, and they lost on
every outcome. The betting apparatus was removed rather than left in place looking
authoritative. The evidence is kept in the repo, because a negative result is worth
keeping.

---

## What it shows

- **Real fixtures** for six European leagues, from the openfootball feed
- **Real form** — each club's last 20 completed matches across this season and last
- **Match-result (1X2) or both-teams-to-score probabilities** from a Poisson model
  fitted per league
- **The model's strongest calls** for the day — the matches it reads most confidently
- **International pricing** for any two national teams you choose

No prices, no slips, no stake sizing. Those came out.

---

## Quick Start

```bash
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`. Fixtures are fetched at run time; no API key needed.

---

## Project Structure

```
├── app.py                        ← The dashboard (run this)
│
├── data/
│   ├── live_fixtures.py          ← Live fixtures + real team form (openfootball)
│   ├── international.py          ← National-team results, ratings and pricing
│   ├── market_data.py            ← Historical matches WITH bookmaker odds
│   ├── apifootball.py            ← APIFootball v3 client (forward odds)
│   ├── odds_link.py              ← Joins an odds feed to a fixture card
│   ├── football_data.py          ← football-data.co.uk ingestion + cache
│   ├── features.py               ← Pre-match feature engineering (leak-free)
│   ├── build_dataset.py          ← CLI: build the labelled training set
│   └── sample_data.py            ← Synthetic demo card (offline fallback)
│
├── models/
│   ├── goal_models.py            ← Poisson / Dixon-Coles / negative binomial / Skellam
│   ├── match_result.py           ← 1X2 pricing from per-league Poisson fits
│   ├── btts_model.py             ← The original heuristic BTTS model
│   ├── daily_picks.py            ← Ranking the card by model confidence
│   ├── evaluate.py               ← Log loss, Brier, calibration, AUC, ROI
│   ├── synthetic.py              ← Match generators with a known process
│   ├── bakeoff.py                ← CLI: score the models head to head
│   └── market_test.py            ← CLI: model vs closing odds — the edge test
│
└── tests/                        ← 78 tests
```

---

## The Evidence

Three tests, in the order they were run. Each one narrowed what the app could
honestly claim.

### 1. Do the models work at all? (`models/bakeoff.py`)

Before trusting a harness on real results, it has to recover a process it was
handed. `models/synthetic.py` generates matches from a stated process:

| Data generated from | Result |
|---|---|
| Poisson | all three tie; negative binomial's `r` hits its ceiling, correctly saying "this is Poisson" |
| Negative binomial (r = 2.5) | negative binomial wins 4/5 runs, recovers r = 3.3 |
| Dixon-Coles (ρ = −0.30) | Dixon-Coles wins 5/5, recovers ρ = −0.30 |

The harness works.

### 2. Which markets have signal? (`--walk-forward`, 154k real matches)

Walk-forward over 8,770 out-of-sample Premier League matches, refitting each season:

| Market | Best model | AUC | Skill vs base rate |
|---|---|---|---|
| **Match result** | Poisson | **0.672** | **+6.9%** |
| Both teams to score | — | 0.513 | **negative** — worse than the base rate |

Team strength predicts *who wins*. It does not predict *whether both teams score* —
a strong side beating a weak one 3-0 rather than 3-1 is close to a coin toss, and
BTTS turns entirely on that coin.

### 3. Does it beat the market? (`models/market_test.py`)

The base rate is a weak opponent. A closing price is not — on this data it is
almost perfectly calibrated (implied 44.6% home wins against 44.7% actual).

Walk-forward across six leagues, 2012 → September 2026, **17,697 out-of-sample
matches**, betting at the best price available:

| Outcome | Model log loss | Market log loss | Model AUC | Market AUC | Bets | ROI | t |
|---|---|---|---|---|---|---|---|
| Home | 0.6265 | **0.5989** | 0.694 | **0.731** | 9,775 | **−4.6%** | −2.91 |
| Draw | 0.5587 | **0.5533** | 0.565 | **0.585** | 5,614 | −0.3% | −0.12 |
| Away | 0.5657 | **0.5411** | 0.701 | **0.739** | 8,887 | −2.9% | −1.24 |

**The model loses on all three, and the home-win loss is statistically significant.**

`roi_t` is the ROI in standard errors from break-even, and it earns its place: on the
Premier League alone the model showed a **+4.1% ROI on draws** — the kind of number
that starts a betting system. Its t was 0.71. Across six leagues it was −0.3%.

```bash
python -m models.market_test --since 2012-01-01
python -m models.bakeoff --data data/processed/btts_dataset.csv --division E0 --walk-forward
python -m models.bakeoff --synthetic negative_binomial --repeats 5
```

---

## Data Sources

| Source | Used for | Notes |
|---|---|---|
| [openfootball/football.json](https://github.com/openfootball/football.json) | Live fixtures + form | Public, no key, auto-updated daily |
| [martj42/international_results](https://github.com/martj42/international_results) | National-team ratings | 49k results since 1872, neutral-venue flags |
| [xgabora/Club-Football-Match-Data](https://github.com/xgabora/Club-Football-Match-Data-2000-2025) | Market test | ~239k matches with closing odds, current to weeks |
| [football-data.co.uk](https://www.football-data.co.uk) | Training datasets | Via a public mirror; the site itself is unreachable from some environments |

### Forward odds

`data/apifootball.py` is a client for [APIFootball v3](https://apiv3.apifootball.com),
the one source tried here that carries **bookmaker odds for matches not yet played**.
Prices are what turn "the model says 62%" into "the model says 62%, the market says
55%" — the difference between a description and a claim.

```bash
export APIFOOTBALL_KEY=your_key
python -m data.apifootball probe --out sample.json     # check the payload shape
python -m data.apifootball odds --from 2026-10-09 --to 2026-10-12
```

**It is unverified against the live API.** It was written in an environment where the
host is unreachable, so requests follow the documented API and parsing is deliberately
tolerant: a renamed field costs one NaN column rather than the run, and `probe` prints
what actually came back. Run `probe` first — its mapping check says how many expected
fields were present.

Once a provider is reachable, `data/odds_link.py` joins its prices to the fixture
card. Feeds disagree on club names — "Manchester Utd", "Manchester United FC" and
"Man United" are one club — so names are normalised and fuzzy-matched, both sides of
a fixture must match, and same-day matching stops a reverse fixture months away from
linking. **Anything still unmatched is reported, not dropped**: a page showing prices
for two thirds of its card with nothing saying which third is worse than one showing
none.

`add_market_comparison()` then puts the model's probability beside the market's and
the gap between them. Read that gap as *how far off consensus a call is* — not as a
signal. When the two disagree the market is usually right: this model lost 4.6%
flat-staking its disagreements, and an independent published model lost 15.2% over
3,834 bets.

Other odds routes tried and closed: football-data.co.uk (historical only), the
football-charts connector (free tier excludes odds — the archive is paid),
the-odds-api / football-data.org / footballdata.io / sofascore (all blocked by the
environment's network policy).

---

## Notes on the Data

Three defects worth knowing about, each found by testing and handled explicitly:

- **openfootball score placeholders.** A minority of matches carry a bare-list score
  instead of the usual object, and across every league-season checked that form is
  *always exactly `[0, 0]`*. Genuine goalless draws appear in the object form, so
  these are placeholders. Counted as results they would invent goalless draws and
  drag every BTTS rate down. They are treated as unknown and reported (141 skipped
  in the current six-league load).
- **Missing kickoff times.** Some leagues publish dates only; those show as `--:--`
  rather than a midnight that looks real.
- **Extrapolated ratings.** Asking the international model for the best side in the
  world against the weakest produced 14 expected goals — not football, and it
  degenerates the scoreline grid. Rates are clamped to a range real matches occupy.

**Leakage discipline.** `data/features.py` makes one chronological pass and reads each
team's history *before* appending the current match to it. `tests/test_dataset.py`
asserts that invariant directly: a leaked feature backtests beautifully, loses money,
and never shows up in the accuracy numbers.

---

## International Matches

Pick **International (manual)** in the sidebar and choose any two national teams.

**Why manual?** No reachable source lists upcoming international fixtures — five were
checked. What exists is history, so the model can be fitted; it just has nothing to
point itself at.

Neutral venues drop the home-advantage term (pairings default to neutral, since a
made-up fixture has no host). Thin data is handled with a date window, time decay,
friendlies at half weight and ridge shrinkage. **Nothing here has been backtested on
international football** — the validation was on club leagues, and the page says so.

---

## Responsible Gambling Note

This dashboard reports probabilities for interest and context. It is not betting
advice, and the evidence above is explicit that these models do not beat bookmaker
prices. If you gamble, do it with money you can afford to lose.

---

## Tech Stack

Python 3.10+ · Streamlit · pandas / NumPy · SciPy (maximum-likelihood fitting)

```bash
python -m pytest tests/ -q      # 78 tests
```
