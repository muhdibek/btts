# ⚽ BTTS Slip AI Dashboard

A **local laptop application** for Both-Teams-To-Score (BTTS) football betting analysis.
Runs entirely on your machine via Streamlit — no cloud, no Telegram, no subscriptions.

---

## Project Structure

```
btts_dashboard/
│
├── app.py                        ← Main Streamlit application (run this)
│
├── data/
│   ├── __init__.py
│   ├── sample_data.py            ← Dashboard fixtures (Phase 1: mock, Phase 2: API)
│   ├── football_data.py          ← football-data.co.uk ingestion + cache
│   ├── features.py               ← Pre-match feature engineering (leak-free)
│   └── build_dataset.py          ← CLI: build the labelled training set
│
├── models/
│   ├── __init__.py
│   ├── btts_model.py             ← Dashboard's heuristic model
│   ├── goal_models.py            ← Poisson / Dixon-Coles / negative binomial / Skellam
│   ├── evaluate.py               ← Log loss, Brier, calibration, ROI
│   ├── synthetic.py              ← Match generators with a known process
│   └── bakeoff.py                ← CLI: score the models head to head
│
├── utils/
│   ├── __init__.py
│   ├── filters.py                ← Match filtering logic
│   ├── slip_generator.py         ← Accumulator slip builder
│   └── api_client.py             ← API-Football integration (Phase 2)
│
├── tests/
│   ├── test_dataset.py           ← Pipeline tests (leakage, Elo, parsing)
│   └── test_bakeoff.py           ← Model tests (recovering a known process)
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# Install packages
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
# From the btts_dashboard/ folder:
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`

---

## Sample Data

`data/sample_data.py` ships a 32-fixture card across the Premier League, La Liga, Serie A,
Bundesliga, Ligue 1 and the Eredivisie, all kicking off **today** (12:00 → ~22:00) so the
target-odds builder always has a realistic same-day pool to work with. Every club appears
once. Swap `load_matches()` for the API client (see Upgrade Path) to run on live fixtures.

---

## How to Use

### Step 1 — Review All Matches (Section 01)
Expand the fixtures table to see all upcoming matches with BTTS probability, odds, and signal labels.

### Step 2 — Use Filtered Matches (Section 02)
The app automatically filters matches based on your **sidebar settings**:
- Minimum BTTS probability (default 62%)
- Attack/defence thresholds
- Odds range

### Step 3 — Generate Slips (Section 03)
Click **⚡ Generate Slips** to build all valid 2-match and 3-match BTTS accumulators.
Slips are ranked by `score = combined_probability × total_odds`.

### Step 3B — Build a Target Odds Slip (Section 03B)
Set **Target Total Odds** in the sidebar (default **25.0**) and click
**🎯 Build 25.00 Odds Slip**.

BTTS prices sit around 1.60–1.95, so a double or treble tops out near 7.0 — reaching a
payout target needs more legs. The builder searches every 2- to 8-leg combination of the
filtered matches and returns the five that reach the target, ranked by:

1. slips that actually clear the target
2. highest combined probability
3. fewest legs
4. tightest to the target

No club appears twice in a slip, and each card shows the combined probability and the
return on your sidebar stake, so a long-odds acca is presented with its real (low)
chance of landing.

### Step 4 — Select Your Slip (Section 04)
Click **Select** on any slip to pin it to the Selected Slip Panel for final review.

---

## Sidebar Controls

| Control | Description |
|---|---|
| Min BTTS Probability | Only show matches above this threshold |
| Min Avg Goals Scored | Attack strength filter |
| Min Avg Goals Conceded | Defensive weakness filter |
| BTTS Odds Range | Acceptable odds window |
| Min Total Slip Odds | Accumulator must exceed this |
| Include 2/3-Match Slips | Toggle slip leg count |
| Max Slips to Show | Cap displayed results |
| Target Total Odds | Payout multiple the target slip must reach (e.g. 25.0) |
| Max Legs in Target Slip | Upper bound on legs used to reach the target (2–8) |
| Stake (units) | Stake used for the potential-return figures |
| Today's fixtures only | Restrict the card to matches kicking off today |
| 🔄 Refresh Data | Clear cache and reload |

---

## Probability Model (Phase 1)

The current model blends three signals:

```
P(BTTS) = 0.40 × P_poisson + 0.45 × P_historical + 0.15 × P_composite
```

| Signal | Weight | Method |
|---|---|---|
| Poisson xG model | 40% | Independent Poisson distributions for each team |
| Historical BTTS rate | 45% | Average of each team's last-20-match BTTS% |
| Attack/defence composite | 15% | Normalised scoring × conceding interaction |

---

## Upgrade Path

### Phase 2A — Training Data Pipeline ✅ built

The dashboard's heuristic model has never been fit to a real outcome. Before any
of that can change, there has to be a labelled history — that pipeline now exists.

```bash
# ten seasons of the big five leagues + Eredivisie
python -m data.build_dataset \
    --leagues E0 SP1 D1 I1 F1 N1 \
    --start-season 2015 --end-season 2024 \
    --out data/processed/btts_dataset.csv

python -m data.build_dataset --list-leagues    # division codes
python -m data.build_dataset --from-cache      # rebuild offline, no downloads
```

**Source:** [football-data.co.uk](https://www.football-data.co.uk) — free CSVs, no API
key, full-time scores (so the BTTS label is exact: `FTHG > 0 & FTAG > 0`) and closing
bookmaker odds in the same row. Season files are cached under `data/raw/`, so a re-run
costs nothing.

**Features** (46, all computed from matches that kicked off *strictly earlier*):

| Group | Columns |
|---|---|
| Form | last-5 / last-10 goals for & against, BTTS rate, failed-to-score rate, clean-sheet rate |
| Venue splits | home side's recent *home* matches, away side's recent *away* matches |
| Elo | pre-match ratings with a goal-difference multiplier, home advantage, and regression toward the mean between seasons |
| Head to head | previous meetings' BTTS rate and average total goals |
| Rest | days since each side's last match |
| Market | overround-free implied probabilities from closing 1X2 and over/under 2.5 prices |

**Leakage discipline.** `data/features.py` makes one chronological pass and reads each
team's history *before* appending the current match to it. `tests/test_dataset.py`
asserts that invariant directly — a leaked feature backtests beautifully and loses
money, and it never shows up in the accuracy numbers.

```bash
python -m pytest tests/ -q
```

Two rules for whatever trains on this:

1. **Split by date, never randomly.** The rows are chronological; shuffling puts future
   matches in the training set and inflates every metric.
2. **Beat the market column, not the base rate.** `mkt_p_over25` already encodes most of
   what the model is trying to learn. A model that beats a coin flip but not the closing
   price has no edge.

### Phase 2A′ — Live Fixtures (API-Football)

1. Sign up at [api-football.com](https://www.api-football.com) (free tier: 100 req/day)
2. Create `.env` file:
   ```
   API_FOOTBALL_KEY=your_key_here
   ```
3. In `data/sample_data.py`, replace `load_matches()` with:
   ```python
   from utils.api_client import fetch_upcoming_fixtures
   return fetch_upcoming_fixtures(league_id=39, season=2024)
   ```

### Phase 2B — XGBoost Model

1. Collect labelled match data (features + BTTS outcome 0/1)
2. Train on `data/processed/btts_dataset.csv`, split by date:
   ```python
   import xgboost as xgb
   model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
   model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30)
   import joblib; joblib.dump(model, "models/btts_xgb_model.pkl")
   ```
3. **Calibrate** on a held-out slice — raw boosted-tree probabilities are overconfident,
   and this app multiplies them across legs, so the error compounds:
   ```python
   from sklearn.calibration import CalibratedClassifierCV
   calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
   calibrated.fit(X_calib, y_calib)
   ```
   Five points of overconfidence per leg (0.65 where the truth is 0.60) makes a 6-leg
   acca read 7.5% when it is really 4.7% — **1.6× overstated**. Note that calibration
   fixes the per-leg bias but not leg *correlation*: `_calculate_combined_probability()`
   takes a straight product, which still overstates a same-day multi.
3. In `models/btts_model.py`, replace `predict_btts_probability()` body:
   ```python
   model = joblib.load("models/btts_xgb_model.pkl")
   features = np.array([row.home_avg_scored, row.away_avg_scored, ...])
   return float(model.predict_proba([features])[0][1])
   ```

### Phase 2C — Historical Data Source
- [football-data.co.uk](https://www.football-data.co.uk) — free CSV files with match results
- Use to build your labelled training set for the XGBoost model

---

## Model Bake-off

Four distributional models, one interface, scored head to head. Every market is
derived from the same scoreline distribution, so the models differ in their
assumption about goals — not in how their output is post-processed.

```
P(BTTS)    = Σ P(H=i, A=j)  for i ≥ 1, j ≥ 1
P(over2.5) = Σ P(H=i, A=j)  for i + j ≥ 3
P(home)    = Σ P(H=i, A=j)  for i > j
```

| Model | Assumption | BTTS? |
|---|---|---|
| `base_rate` | every match gets the training base rate | the floor to beat |
| `team_rate` | mean of the two sides' historical BTTS rates | the dashboard's strongest heuristic term, isolated |
| `poisson` | independent Poisson, log-linear attack/defence/home advantage, fitted by MLE | yes |
| `dixon_coles` | Poisson + the low-score correction on the 0-0, 1-0, 0-1 and 1-1 cells, + optional time decay | yes |
| `negative_binomial` | same means, variance λ + λ²/r, for overdispersed goals | yes |
| `skellam` | models the goal *difference* | **no** — reports NaN |

Skellam is the honest exception. A difference distribution cannot say whether
both teams scored, because 0-0 and 1-1 are the same difference. It prices 1X2
and handicaps; on BTTS it reports NaN and the bake-off leaves it unscored
rather than substituting a guess.

### Running it

```bash
python -m models.bakeoff --synthetic poisson            # validate on known truth
python -m models.bakeoff --synthetic negative_binomial --repeats 5
python -m models.bakeoff --data data/processed/btts_dataset.csv --calibration
python -m models.bakeoff --synthetic dixon_coles --market p_draw
```

The split is always by **date**. On synthetic data the table also carries an
**ORACLE** row — the generating process's own probabilities — which is the
ceiling no model can beat and the measure of how much remaining loss is
irreducible noise rather than model error.

### Validating the harness

Before trusting a bake-off on real results, it has to recover a process it was
given. `models/synthetic.py` generates matches from a stated process, and
`tests/test_bakeoff.py` asserts the harness gets each one right:

| Data generated from | Expected result | Observed (5 leagues each) |
|---|---|---|
| Poisson | Poisson wins; negative binomial ties it by fitting r → ∞ | all three tie at 0.638 log loss; fitted r hits its 500 ceiling |
| Negative binomial (r = 2.5) | negative binomial wins | wins 4/5, mean log loss 0.666 vs Poisson 0.700; recovered r = 3.3 |
| Dixon-Coles (ρ = −0.30) | Dixon-Coles wins | wins 5/5, log loss 0.625 vs Poisson 0.628; recovered ρ = −0.30 |

Two findings worth carrying into any real run:

- **Single runs are noise.** At ~285 test matches the standard deviation of log
  loss across seeds is ±0.01 — larger than the gap between most of these
  models. A model can even score below the oracle on a lucky sample. Use
  `--repeats` and read the win counts, not one table.
- **The Dixon-Coles correction is small.** At ρ = −0.15 over 1,140 matches it is
  undetectable on BTTS; it only separates cleanly at ρ = −0.30 with 3,000+
  matches, and it shows up most strongly on the draw market rather than BTTS.
- **"Variance > mean" does not prove overdispersion.** Poisson-generated
  matches here show variance/mean ≈ 1.18, because λ varies between matches.
  Only a fitted r says whether goals are genuinely overdispersed.

---

## Responsible Gambling Note

This dashboard is a **probability analysis tool** for research and educational purposes.
No model guarantees outcomes. Always gamble responsibly within your means.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **Pandas / NumPy** — data manipulation
- **Scikit-learn** — Phase 1 utilities
- **XGBoost** — Phase 2 model
- **Requests + python-dotenv** — API integration
- **itertools** — combination generation
