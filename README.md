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
│   └── sample_data.py            ← Match data loader (Phase 1: mock, Phase 2: API)
│
├── models/
│   ├── __init__.py
│   └── btts_model.py             ← BTTS probability model (Phase 1: heuristic, Phase 2: XGBoost)
│
├── utils/
│   ├── __init__.py
│   ├── filters.py                ← Match filtering logic
│   ├── slip_generator.py         ← Accumulator slip builder
│   └── api_client.py             ← API-Football integration (Phase 2)
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

### Phase 2A — Real API Data

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
2. Train:
   ```python
   import xgboost as xgb
   model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05)
   model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=30)
   import joblib; joblib.dump(model, "models/btts_xgb_model.pkl")
   ```
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
