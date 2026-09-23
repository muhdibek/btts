"""
app.py
------
BTTS Slip AI Dashboard — Main Streamlit Application

Run with:
    streamlit run app.py

Layout:
  [Sidebar]    Filter controls
  [Section 1]  All Matches Table
  [Section 2]  Filtered High-Probability Matches
  [Section 3]  Generated Slips
  [Section 3B] Target Odds Slip (e.g. a 25.00 acca from today's card)
  [Section 4]  Selected Slip Panel
"""

import sys
import os
from pathlib import Path

# Ensure local modules are importable regardless of how/where Streamlit launches.
# os.path.dirname(__file__) can be empty string in some environments — abspath fixes this.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from data.sample_data       import load_matches
from data.live_fixtures     import (DEFAULT_LEAGUES, LEAGUES as LIVE_LEAGUES,
                                   available_dates, build_fixture_card, fair_odds,
                                   feed_status, load_leagues, malformed_score_count,
                                   next_matchday)
from models.btts_model      import score_matches, get_confidence_label
from utils.filters          import (filter_high_probability_matches, filter_todays_matches,
                                   get_filter_summary)
from utils.slip_generator   import (generate_slips, slips_to_dataframe,
                                   build_target_odds_slips, potential_return)


# ============================================================
#  PAGE CONFIG  (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title   = "BTTS Slip AI Dashboard",
    page_icon    = "⚽",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)


# ============================================================
#  HTML HELPER
# ============================================================
def render_html(html: str) -> None:
    """
    Render a raw HTML block.

    Streamlit runs the string through Markdown first, and Markdown turns any
    line indented by 4+ spaces into a code block — which is why HTML written
    inside an indented f-string shows up as literal tags on the page. Stripping
    the indentation off every line keeps the markup rendering as markup.
    """
    cleaned = "\n".join(line.strip() for line in html.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)



# ============================================================
#  CUSTOM CSS — Dark tactical theme
# ============================================================
render_html("""
<style>
/* ---------- Root palette ---------- */
:root {
    --bg-primary:   #0a0e1a;
    --bg-card:      #111827;
    --bg-card2:     #1a2236;
    --accent:       #00e5ff;
    --accent2:      #7c3aed;
    --green:        #00ff88;
    --yellow:       #f5d020;
    --red:          #ff4444;
    --text-primary: #e8eaf0;
    --text-muted:   #8892a4;
    --border:       #1e2d45;
}

/* ---------- App background ---------- */
.stApp {
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: 'Courier New', monospace;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #080c16 !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSlider label {
    color: var(--text-muted) !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---------- Section headers ---------- */
.section-header {
    background: linear-gradient(90deg, var(--accent2) 0%, transparent 100%);
    padding: 8px 16px;
    border-left: 3px solid var(--accent);
    margin: 24px 0 12px 0;
    border-radius: 2px;
}
.section-header h3 {
    color: var(--accent);
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin: 0;
}

/* ---------- Metric cards ---------- */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: 6px;
    padding: 14px 20px;
    min-width: 140px;
    flex: 1;
}
.metric-card .m-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-card .m-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* ---------- Match table ---------- */
.match-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    margin-bottom: 8px;
}
.match-table th {
    background: #0d1525;
    color: var(--text-muted);
    text-transform: uppercase;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    padding: 10px 12px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.match-table td {
    padding: 10px 12px;
    border-bottom: 1px solid #161e2e;
    color: var(--text-primary);
    vertical-align: middle;
}
.match-table tr:hover td { background: #141c2e; }

/* ---------- Probability badges ---------- */
.prob-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.8rem;
    letter-spacing: 0.03em;
}

/* ---------- Slip cards ---------- */
.slip-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 12px;
    transition: all 0.2s;
    cursor: pointer;
}
.slip-card:hover { border-color: var(--accent2); }
.slip-card.selected {
    border-color: var(--accent);
    background: #0d1e30;
    box-shadow: 0 0 0 1px var(--accent), inset 0 0 20px rgba(0,229,255,0.04);
}
.slip-card .slip-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.slip-card .slip-id {
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.1em;
}
.slip-card .slip-score {
    font-size: 0.75rem;
    color: var(--accent);
    font-weight: 700;
}
.slip-card .match-leg {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #1a2438;
    font-size: 0.82rem;
}
.slip-card .match-leg:last-child { border-bottom: none; }
.slip-footer {
    display: flex;
    gap: 20px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #1a2438;
    font-size: 0.75rem;
    color: var(--text-muted);
}
.slip-footer .sf-val { color: var(--text-primary); font-weight: 700; }

/* ---------- Selected slip panel ---------- */
.selected-panel {
    background: linear-gradient(135deg, #0d1e2e 0%, #0a1628 100%);
    border: 1px solid var(--accent);
    border-radius: 10px;
    padding: 24px;
    box-shadow: 0 0 30px rgba(0,229,255,0.08);
}
.selected-panel h4 {
    color: var(--accent);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 16px;
}

/* ---------- Streamlit overrides ---------- */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    padding: 8px 20px !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6d28d9, var(--accent2)) !important;
    box-shadow: 0 0 12px rgba(124,58,237,0.4) !important;
}

div[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent2);
    padding: 12px;
    border-radius: 6px;
}
div[data-testid="stMetric"] label { color: var(--text-muted) !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--accent) !important; }

/* Selectbox */
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-size: 0.82rem !important;
}

/* Scrollbars */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""")


# ============================================================
#  SESSION STATE INIT
# ============================================================
if "selected_slip_id" not in st.session_state:
    st.session_state.selected_slip_id = None
if "slips" not in st.session_state:
    st.session_state.slips = []
if "slips_generated" not in st.session_state:
    st.session_state.slips_generated = False
if "target_slips" not in st.session_state:
    st.session_state.target_slips = []
if "target_generated" not in st.session_state:
    st.session_state.target_generated = False


# ============================================================
#  SIDEBAR — filter controls
# ============================================================
with st.sidebar:
    render_html("""
    <div style="text-align:center; padding: 16px 0 24px 0;">
        <div style="font-size:2rem;">⚽</div>
        <div style="font-size:0.7rem; color:#00e5ff; letter-spacing:0.2em;
                    text-transform:uppercase; font-weight:700;">BTTS Slip AI</div>
        <div style="font-size:0.6rem; color:#8892a4; margin-top:4px;">
            Local Betting Dashboard
        </div>
    </div>
    """)

    st.markdown("---")

    # --- Data source ---
    st.markdown("**📡 Data Source**")
    source = st.radio(
        "Fixtures",
        ["Live fixtures", "Sample data"],
        index=0,
        help="Live: real fixtures and real team form from the openfootball feed. "
             "Sample: the synthetic demo card.",
        label_visibility="collapsed",
    )
    live_mode = source == "Live fixtures"
    # Live feed has no bookmaker prices, so odds shown are 1 / probability.
    odds_label = "Fair Odds" if live_mode else "BTTS Odds"

    if live_mode:
        live_league_codes = st.multiselect(
            "Leagues",
            options=list(LIVE_LEAGUES),
            default=DEFAULT_LEAGUES,
            format_func=lambda code: LIVE_LEAGUES[code],
        )

    st.markdown("---")

    # --- Filter thresholds ---
    st.markdown("**🎯 Probability Filter**")
    min_btts_prob = st.slider(
        "Min BTTS Probability (%)",
        min_value=40, max_value=90, value=55, step=1,
        help="Only show matches where BTTS probability ≥ this value"
    ) / 100.0

    st.markdown("**📊 Attack / Defence**")
    min_avg_scored = st.slider(
        "Min Avg Goals Scored",
        min_value=0.5, max_value=2.5, value=1.1, step=0.1
    )
    max_avg_conceded = st.slider(
        "Min Avg Goals Conceded (weakness)",
        min_value=0.8, max_value=2.5, value=1.3, step=0.1
    )

    st.markdown(f"**💰 {odds_label} Range**")
    odds_range = st.slider(
        f"{odds_label} Range",
        min_value=1.3, max_value=3.0,
        value=(1.30, 3.00) if live_mode else (1.55, 2.20), step=0.05,
        help=("Fair odds are 1 / probability, so this range is just a probability "
              "window in disguise — a low ceiling here removes your most likely "
              "matches, not your worst ones.") if live_mode
             else "Acceptable bookmaker price window",
    )

    st.markdown("**🎰 Slip Settings**")
    min_total_odds = st.slider(
        "Min Total Slip Odds",
        min_value=2.0, max_value=6.0, value=3.0, step=0.1
    )
    include_doubles  = st.checkbox("Include 2-Match Slips",  value=True)
    include_trebles  = st.checkbox("Include 3-Match Slips",  value=True)
    max_slips_shown  = st.slider("Max Slips to Show", 5, 30, 15)

    st.markdown("**🎯 Target Odds Slip**")
    target_odds = st.number_input(
        "Target Total Odds",
        min_value=2.0, max_value=200.0, value=25.0, step=1.0,
        help="Payout multiple the accumulator must reach — e.g. 25.0 for a 25 odds slip"
    )
    max_target_legs = st.slider(
        "Max Legs in Target Slip",
        min_value=2, max_value=8, value=8,
        help="BTTS prices around 1.6–1.9 usually need 5–7 legs to reach 25.0"
    )
    stake = st.number_input("Stake (units)", min_value=1.0, max_value=1000.0, value=10.0, step=1.0)

    if not live_mode:
        st.markdown("**📅 Match Day**")
        today_only = st.checkbox("Today's fixtures only", value=True)
    else:
        today_only = False

    st.markdown("---")
    if st.button("🔄  Refresh Data"):
        st.cache_data.clear()
        st.session_state.slips = []
        st.session_state.slips_generated = False
        st.session_state.target_slips = []
        st.session_state.target_generated = False
        st.session_state.selected_slip_id = None
        st.rerun()


# ============================================================
#  DATA PIPELINE
# ============================================================
@st.cache_data(ttl=300)
def get_scored_matches() -> pd.DataFrame:
    """Load and score the synthetic demo card (cached for 5 minutes)."""
    return score_matches(load_matches())


@st.cache_data(ttl=1800)
def get_live_season(codes: tuple[str, ...]) -> pd.DataFrame:
    """
    Every fixture and result for the chosen leagues, this season and last.

    Last season is included so team form is not built from three August
    matches, and it is cached for half an hour — the feed updates daily.
    """
    return load_leagues(["2026-27", "2025-26"], codes=list(codes))


def add_fair_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing prices with fair odds — 1 / probability, no margin.

    The live feed carries no bookmaker odds. Rather than invent prices, the
    break-even price for the model's own probability is used, and labelled as
    fair everywhere it appears. A real book prices below this, never above, so
    any "return" computed from these is an optimistic ceiling and not a payout.
    """
    df = df.copy()
    missing = df["btts_odds"].isna()
    if missing.any():
        df.loc[missing, "btts_odds"] = [fair_odds(p) for p in df.loc[missing, "btts_prob"]]
    return df


# ---- Load data ----------------------------------------------------------
odds_are_fair = False
live_meta: dict = {}

if live_mode:
    season_df = get_live_season(tuple(live_league_codes))

    if season_df.empty:
        st.error("The live fixture feed returned nothing — it may be unreachable "
                 "from here. Switch to Sample data in the sidebar to keep working.")
        st.stop()

    upcoming = available_dates(season_df)
    default_day = next_matchday(season_df)

    with st.sidebar:
        st.markdown("**📅 Matchday**")
        if upcoming:
            chosen_day = st.selectbox(
                "Date",
                options=upcoming,
                index=upcoming.index(default_day) if default_day in upcoming else 0,
                format_func=lambda d: d.strftime("%a %d %b %Y"),
                label_visibility="collapsed",
            )
        else:
            chosen_day = None
            st.caption("No upcoming fixtures in the feed.")

    card_df = build_fixture_card(season_df, on_date=chosen_day)

    if card_df.empty:
        render_html(f"""
        <div style="background:#1a1400; border:1px solid #f5d02044; border-radius:6px;
                    padding:16px; color:#f5d020; font-size:0.85rem;">
            No fixtures on {chosen_day:%A %d %B %Y} in the selected leagues.
            {"Next matchday: <strong>%s</strong>." % default_day.strftime("%A %d %B")
             if default_day else ""}
        </div>
        """)
        st.stop()

    all_matches_df = add_fair_odds(score_matches(card_df))
    odds_are_fair  = True
    live_meta = {
        "day":        chosen_day,
        "leagues":    len(live_league_codes),
        "results":    int(season_df["played"].sum()),
        "unusable":   malformed_score_count(season_df),
        "fetched":    feed_status(["2026-27", "2025-26"], live_league_codes).get("last_fetched"),
    }
else:
    all_matches_df = get_scored_matches()
    if today_only:
        all_matches_df = filter_todays_matches(all_matches_df)

# Apply filters
slip_sizes = []
if include_doubles: slip_sizes.append(2)
if include_trebles: slip_sizes.append(3)

filtered_df = filter_high_probability_matches(
    df               = all_matches_df,
    min_btts_prob    = min_btts_prob,
    min_avg_scored   = min_avg_scored,
    max_avg_conceded = max_avg_conceded,
    min_btts_odds    = odds_range[0],
    max_btts_odds    = odds_range[1],
)

summary = get_filter_summary(all_matches_df, filtered_df)


# ============================================================
#  PAGE HEADER
# ============================================================
col_title, col_status = st.columns([3, 1])
with col_title:
    render_html(f"""
    <div style="padding: 8px 0 4px 0;">
        <span style="font-size:1.6rem; font-weight:800; color:#e8eaf0;
                     letter-spacing:0.05em;">BTTS SLIP</span>
        <span style="font-size:1.6rem; font-weight:800; color:#00e5ff;
                     letter-spacing:0.05em;"> AI DASHBOARD</span>
        <div style="font-size:0.7rem; color:#8892a4; margin-top:2px;
                    text-transform:uppercase; letter-spacing:0.15em;">
            Both Teams To Score · Local Intelligence ·
            {"Live Fixtures" if live_mode else "Synthetic Demo Card"}
        </div>
    </div>
    """)
with col_status:
    if live_mode:
        chip_bg, chip_border, chip_text = "#0d2518", "#00ff88", "#00ff88"
        chip_label = (f"● LIVE FEED · {live_meta['day']:%a %d %b} · "
                      f"{len(all_matches_df)} Fixtures")
    else:
        chip_bg, chip_border, chip_text = "#1a1400", "#f5d020", "#f5d020"
        chip_label = f"◌ SAMPLE DATA · {len(all_matches_df)} Fixtures"

    render_html(f"""
    <div style="text-align:right; padding-top:12px;">
        <span style="background:{chip_bg}; border:1px solid {chip_border}; color:{chip_text};
                     padding:4px 10px; border-radius:20px; font-size:0.68rem;
                     letter-spacing:0.1em; text-transform:uppercase;">
            {chip_label}
        </span>
    </div>
    """)

render_html("<div style='margin-bottom:4px;'></div>")

# --- Top KPI row ---
k1, k2, k3, k4, k5 = st.columns(5)
kpis = [
    (k1, str(summary["total_matches"]),     "Total Fixtures"),
    (k2, str(summary["filtered_matches"]),  "BTTS Candidates"),
    (k3, f"{summary['avg_btts_prob']}%",    "Avg BTTS Prob"),
    (k4, f"{summary['max_btts_prob']}%",    "Best Match Prob"),
    (k5, str(len(st.session_state.slips)),  "Slips Generated"),
]
for col, val, label in kpis:
    with col:
        st.metric(label=label, value=val)


# ============================================================
#  DATA PROVENANCE / HONESTY BANNER
# ============================================================
if live_mode:
    fetched = live_meta.get("fetched")
    render_html(f"""
    <div style="background:#0d1525; border:1px solid #1e2d45; border-left:2px solid #00e5ff;
                border-radius:6px; padding:12px 16px; margin:8px 0 4px 0;
                font-size:0.72rem; color:#8892a4; line-height:1.6;">
        <strong style="color:#00e5ff;">Real fixtures.</strong>
        {live_meta['leagues']} leagues · team form built from
        <strong style="color:#e8eaf0;">{live_meta['results']:,}</strong> completed matches
        (this season and last){f" · feed fetched {fetched:%d %b %H:%M}" if fetched is not None else ""}
        {f" · {live_meta['unusable']} matches skipped for unusable scores" if live_meta['unusable'] else ""}.
        <br>
        <strong style="color:#f5d020;">Odds below are FAIR odds (1 / probability), not
        bookmaker prices</strong> — this feed carries no market odds. A real book pays less,
        so any return shown is a ceiling, not a payout.
        <br>
        <strong style="color:#ff9d00;">These probabilities have no measured edge.</strong>
        Backtested on 154k historical matches, this heuristic and four statistical models all
        scored worse than simply using the league BTTS base rate. Treat the numbers as
        descriptive, not predictive — see the README bake-off results.
    </div>
    """)

# ============================================================
#  SECTION 1: ALL MATCHES TABLE
# ============================================================
render_html("""
<div class="section-header"><h3>01 · All Upcoming Fixtures</h3></div>
""")

def render_matches_table(df: pd.DataFrame, highlight_threshold: float = 0.62):
    """Render the match list as a styled HTML table."""
    rows_html = ""
    for _, row in df.iterrows():
        prob        = row["btts_prob"]
        label, color = get_confidence_label(prob)
        prob_pct    = f"{prob*100:.1f}%"
        
        # Row background for high-prob matches
        row_style = "background: #0d1a2e;" if prob >= highlight_threshold else ""
        
        rows_html += f"""
        <tr style="{row_style}">
            <td style="color:#8892a4; font-size:0.7rem;">{row['kickoff']}</td>
            <td><strong>{row['home_team']}</strong>
                <span style="color:#8892a4; font-size:0.72rem;"> vs </span>
                <strong>{row['away_team']}</strong></td>
            <td style="color:#8892a4;">{row['league']}</td>
            <td>
                <span class="prob-badge" style="background:{color}22; color:{color};
                      border:1px solid {color}44;">{prob_pct}</span>
            </td>
            <td style="font-size:0.75rem; color:#8892a4;">{label}</td>
            <td style="color:#f5d020; font-weight:700;">{row['btts_odds']}</td>
            <td style="color:#8892a4; font-size:0.75rem;">
                {row['home_avg_scored']} / {row['away_avg_scored']}
            </td>
            <td style="color:#8892a4; font-size:0.75rem;">
                {row['home_avg_conceded']} / {row['away_avg_conceded']}
            </td>
        </tr>
        """

    table_html = f"""
    <table class="match-table">
        <thead>
            <tr>
                <th>Kickoff</th>
                <th>Match</th>
                <th>League</th>
                <th>BTTS Prob</th>
                <th>Signal</th>
                <th>{odds_label}</th>
                <th>Avg Scored H/A</th>
                <th>Avg Conceded H/A</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """
    render_html(table_html)


with st.expander("View All Matches", expanded=False):
    render_matches_table(all_matches_df)


# ============================================================
#  SECTION 2: FILTERED HIGH-PROBABILITY MATCHES
# ============================================================
render_html("""
<div class="section-header"><h3>02 · High Probability BTTS Candidates</h3></div>
""")

if filtered_df.empty:
    render_html("""
    <div style="background:#1a0a0a; border:1px solid #ff444444; border-radius:6px;
                padding:16px; color:#ff7777; font-size:0.82rem; text-align:center;">
        ⚠️ No matches pass your current filters. Try lowering the BTTS threshold in the sidebar.
    </div>
    """)
else:
    render_html(
        f"<div style='font-size:0.75rem; color:#8892a4; margin-bottom:8px;'>"
        f"Showing <strong style='color:#00e5ff;'>{len(filtered_df)}</strong> of "
        f"{len(all_matches_df)} fixtures · Filter: BTTS ≥ {min_btts_prob*100:.0f}%"
        f"</div>")
    render_matches_table(filtered_df, highlight_threshold=0.0)


# ============================================================
#  SECTION 3: SLIP GENERATOR
# ============================================================
render_html("""
<div class="section-header"><h3>03 · Slip Generator</h3></div>
""")

gen_col, info_col = st.columns([1, 3])

with gen_col:
    generate_clicked = st.button("⚡  Generate Slips")

with info_col:
    render_html(
        f"<div style='font-size:0.75rem; color:#8892a4; padding-top:8px;'>"
        f"Generates {'2-match + ' if include_doubles else ''}"
        f"{'3-match ' if include_trebles else ''}combinations · "
        f"Min total odds: <strong style='color:#f5d020;'>{min_total_odds:.1f}</strong> · "
        f"Ranked by: prob × odds"
        f"</div>")

# Generate slips on button click
if generate_clicked:
    if filtered_df.empty:
        st.warning("No filtered matches to build slips from. Adjust your filters first.")
    elif not slip_sizes:
        st.warning("Enable at least one slip size (doubles or trebles) in the sidebar.")
    else:
        st.session_state.slips = generate_slips(
            filtered_df    = filtered_df,
            min_total_odds = min_total_odds,
            max_slips      = max_slips_shown,
            slip_sizes     = tuple(slip_sizes),
        )
        st.session_state.slips_generated = True
        st.session_state.selected_slip_id = None

# Display slips
if st.session_state.slips_generated:
    slips = st.session_state.slips

    if not slips:
        render_html("""
        <div style="background:#1a1400; border:1px solid #f5d02044; border-radius:6px;
                    padding:16px; color:#f5d020; font-size:0.82rem; text-align:center;">
            ⚠️ No slips meet the minimum total odds threshold.
            Try lowering the "Min Total Slip Odds" slider or enabling more matches.
        </div>
        """)
    else:
        render_html(
            f"<div style='font-size:0.75rem; color:#8892a4; margin-bottom:12px;'>"
            f"Found <strong style='color:#00ff88;'>{len(slips)}</strong> qualifying slips"
            f"</div>")

        # Render each slip as a card with a select button
        for slip in slips:
            is_selected = (st.session_state.selected_slip_id == slip["slip_id"])
            selected_class = "selected" if is_selected else ""

            # Build match legs HTML
            legs_html = ""
            for match, prob, odds in zip(
                slip["matches"], slip["btts_probs"], slip["btts_odds"]
            ):
                _, col = get_confidence_label(prob / 100)
                legs_html += f"""
                <div class="match-leg">
                    <span>{match}</span>
                    <span style="display:flex; gap:12px; align-items:center;">
                        <span class="prob-badge"
                              style="background:{col}22; color:{col};
                                     border:1px solid {col}44; font-size:0.73rem;">
                            {prob}%
                        </span>
                        <span style="color:#f5d020; font-weight:700;
                                     font-size:0.8rem;">{odds}</span>
                    </span>
                </div>
                """

            # Odds colour: green if ≥ 4, yellow if ≥ 3
            odds_color = "#00ff88" if slip["total_odds"] >= 4.0 else "#f5d020"

            selected_badge = (
                '<span style="background:#00e5ff22; border:1px solid #00e5ff; '
                'color:#00e5ff; padding:2px 8px; border-radius:20px; '
                'font-size:0.65rem; letter-spacing:0.1em;">✓ SELECTED</span>'
                if is_selected else ""
            )

            card_html = f"""
            <div class="slip-card {selected_class}">
                <div class="slip-header">
                    <div>
                        <span class="slip-id">{slip['slip_id']}</span>
                        &nbsp;&nbsp;
                        <span style="background:#1a1030; border:1px solid #7c3aed44;
                              color:#a78bfa; padding:2px 8px; border-radius:20px;
                              font-size:0.65rem;">{slip['legs']}-Leg Acca</span>
                        &nbsp;{selected_badge}
                    </div>
                    <span class="slip-score">Score: {slip['slip_score']:.3f}</span>
                </div>
                {legs_html}
                <div class="slip-footer">
                    <div>Combined Prob <span class="sf-val">{slip['combined_prob']}%</span></div>
                    <div>{"Fair Total" if live_mode else "Total Odds"} <span class="sf-val" style="color:{odds_color};">
                        {slip['total_odds']}</span></div>
                    <div>Matches <span class="sf-val">{slip['legs']}</span></div>
                </div>
            </div>
            """
            render_html(card_html)

            # Select button per slip
            btn_label = "✓ Selected" if is_selected else f"Select  {slip['slip_id']}"
            if st.button(btn_label, key=f"btn_{slip['slip_id']}"):
                st.session_state.selected_slip_id = slip["slip_id"]
                st.rerun()


# ============================================================
#  SECTION 3B: TARGET ODDS SLIP
# ============================================================
render_html("""
<div class="section-header"><h3>03B · Target Odds Slip</h3></div>
""")

tgt_col, tgt_info_col = st.columns([1, 3])

with tgt_col:
    target_clicked = st.button(f"🎯  Build {target_odds:.2f} Odds Slip")

with tgt_info_col:
    render_html(
        f"<div style='font-size:0.75rem; color:#8892a4; padding-top:8px;'>"
        f"Searches 2–{max_target_legs} leg combinations for the "
        f"<strong style='color:#f5d020;'>highest-probability</strong> acca that reaches "
        f"<strong style='color:#f5d020;'>{target_odds:.2f}</strong> · "
        f"no club used twice"
        f"</div>")

if target_clicked:
    if filtered_df.empty:
        st.warning("No filtered matches to build a target slip from. Adjust your filters first.")
    else:
        st.session_state.target_slips = build_target_odds_slips(
            filtered_df = filtered_df,
            target_odds = target_odds,
            min_legs    = 2,
            max_legs    = max_target_legs,
            max_slips   = 5,
        )
        st.session_state.target_generated = True
        st.session_state.selected_slip_id = None

if st.session_state.target_generated:
    target_slips = st.session_state.target_slips

    if not target_slips:
        best_possible = 1.0
        for o in sorted(filtered_df["btts_odds"], reverse=True)[:max_target_legs]:
            best_possible *= o
        render_html(f"""
        <div style="background:#1a1400; border:1px solid #f5d02044; border-radius:6px;
                    padding:16px; color:#f5d020; font-size:0.82rem;">
            ⚠️ No combination of today's filtered matches reaches {target_odds:.2f}.
            The best {max_target_legs}-leg acca available is
            <strong>{best_possible:.2f}</strong>.<br>
            <span style="font-size:0.75rem;">
                Widen the BTTS odds range, lower the probability filter, or raise the max leg
                count in the sidebar to open up more candidates.
            </span>
        </div>
        """)
    else:
        render_html(
            f"<div style='font-size:0.75rem; color:#8892a4; margin-bottom:12px;'>"
            f"Best <strong style='color:#00ff88;'>{len(target_slips)}</strong> slips for a "
            f"<strong style='color:#f5d020;'>{target_odds:.2f}</strong> target"
            f"</div>")

        for slip in target_slips:
            is_selected    = (st.session_state.selected_slip_id == slip["slip_id"])
            selected_class = "selected" if is_selected else ""

            legs_html = ""
            for match, prob, odds, kickoff in zip(
                slip["matches"], slip["btts_probs"], slip["btts_odds"], slip["kickoffs"]
            ):
                _, col = get_confidence_label(prob / 100)
                legs_html += f"""
                <div class="match-leg">
                    <span>{match}
                        <span style="color:#8892a4; font-size:0.68rem;">
                            &nbsp;{kickoff[11:]}
                        </span>
                    </span>
                    <span style="display:flex; gap:12px; align-items:center;">
                        <span class="prob-badge"
                              style="background:{col}22; color:{col};
                                     border:1px solid {col}44; font-size:0.73rem;">
                            {prob}%
                        </span>
                        <span style="color:#f5d020; font-weight:700;
                                     font-size:0.8rem;">{odds}</span>
                    </span>
                </div>
                """

            reach_color = "#00ff88" if slip["clears_target"] else "#f5d020"
            reach_label = (
                f"✓ Clears target (+{slip['odds_gap']:.2f})" if slip["clears_target"]
                else f"Under target ({slip['odds_gap']:.2f})"
            )

            render_html(f"""
            <div class="slip-card {selected_class}">
                <div class="slip-header">
                    <div>
                        <span class="slip-id">{slip['slip_id']}</span>
                        &nbsp;&nbsp;
                        <span style="background:#1a1030; border:1px solid #7c3aed44;
                              color:#a78bfa; padding:2px 8px; border-radius:20px;
                              font-size:0.65rem;">{slip['legs']}-Leg Acca</span>
                        &nbsp;
                        <span style="background:{reach_color}22; border:1px solid {reach_color}44;
                              color:{reach_color}; padding:2px 8px; border-radius:20px;
                              font-size:0.65rem;">{reach_label}</span>
                    </div>
                    <span class="slip-score">Score: {slip['slip_score']:.3f}</span>
                </div>
                {legs_html}
                <div class="slip-footer">
                    <div>Combined Prob <span class="sf-val">{slip['combined_prob']}%</span></div>
                    <div>{"Fair Total" if live_mode else "Total Odds"} <span class="sf-val" style="color:{reach_color};">
                        {slip['total_odds']}</span></div>
                    <div>Returns {stake:.0f}u →
                        <span class="sf-val">{potential_return(slip['total_odds'], stake):.2f}u</span>
                    </div>
                </div>
            </div>
            """)

            btn_label = "✓ Selected" if is_selected else f"Select  {slip['slip_id']}"
            if st.button(btn_label, key=f"btn_{slip['slip_id']}"):
                st.session_state.selected_slip_id = slip["slip_id"]
                st.rerun()

        render_html(
            "<div style='font-size:0.7rem; color:#8892a4; margin-top:4px;'>"
            "A slip this long is a low-probability bet by design — the combined probability "
            "above is the model's honest estimate of it landing."
            "</div>")


# ============================================================
#  SECTION 4: SELECTED SLIP PANEL
# ============================================================
render_html("""
<div class="section-header"><h3>04 · Selected Slip</h3></div>
""")

if st.session_state.selected_slip_id is None:
    render_html("""
    <div style="background:#0d1525; border:1px dashed #1e2d45; border-radius:8px;
                padding:24px; text-align:center; color:#8892a4; font-size:0.82rem;">
        No slip selected yet.<br>
        <span style="font-size:0.7rem;">Generate slips above and click "Select" on your preferred combination.</span>
    </div>
    """)
else:
    # Find the selected slip object
    selected = next(
        (s for s in (st.session_state.slips + st.session_state.target_slips)
         if s["slip_id"] == st.session_state.selected_slip_id),
        None
    )

    if selected:
        odds_color = "#00ff88" if selected["total_odds"] >= 4.0 else "#f5d020"

        legs_detail = ""
        for i, (match, prob, odds, kickoff) in enumerate(zip(
            selected["matches"],
            selected["btts_probs"],
            selected["btts_odds"],
            selected["kickoffs"],
        ), 1):
            _, col = get_confidence_label(prob / 100)
            legs_detail += f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:10px 0; border-bottom:1px solid #1a2438;">
                <div>
                    <div style="font-size:0.65rem; color:#8892a4;
                                text-transform:uppercase; letter-spacing:0.08em;">
                        Leg {i} · {kickoff}
                    </div>
                    <div style="font-size:0.88rem; font-weight:700;
                                color:#e8eaf0; margin-top:2px;">{match}</div>
                    <div style="font-size:0.7rem; color:#8892a4; margin-top:2px;">
                        Market: Both Teams To Score
                    </div>
                </div>
                <div style="text-align:right;">
                    <div class="prob-badge"
                         style="background:{col}22; color:{col};
                                border:1px solid {col}44;">{prob}%</div>
                    <div style="color:#f5d020; font-size:1.1rem; font-weight:800;
                                margin-top:4px;">{odds}</div>
                </div>
            </div>
            """

        panel_html = f"""
        <div class="selected-panel">
            <div style="display:flex; justify-content:space-between;
                        align-items:flex-start; margin-bottom:16px;">
                <div>
                    <h4>📋 Your Selected Slip</h4>
                    <div style="font-size:0.7rem; color:#8892a4;">{selected['slip_id']} ·
                        {selected['legs']}-Leg BTTS Accumulator</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:0.65rem; color:#8892a4;
                                text-transform:uppercase;">{"Fair Total Odds" if live_mode else "Total Odds"}</div>
                    <div style="font-size:2rem; font-weight:800;
                                color:{odds_color}; line-height:1;">{selected['total_odds']}</div>
                </div>
            </div>

            {legs_detail}

            <div style="display:flex; gap:32px; margin-top:16px; padding-top:12px;
                        border-top:1px solid #1e2d45;">
                <div>
                    <div style="font-size:0.65rem; color:#8892a4;
                                text-transform:uppercase;">Combined BTTS Prob</div>
                    <div style="font-size:1.3rem; font-weight:700;
                                color:#00ff88;">{selected['combined_prob']}%</div>
                </div>
                <div>
                    <div style="font-size:0.65rem; color:#8892a4;
                                text-transform:uppercase;">Slip Score</div>
                    <div style="font-size:1.3rem; font-weight:700;
                                color:#00e5ff;">{selected['slip_score']:.3f}</div>
                </div>
                <div>
                    <div style="font-size:0.65rem; color:#8892a4;
                                text-transform:uppercase;">Legs</div>
                    <div style="font-size:1.3rem; font-weight:700;
                                color:#a78bfa;">{selected['legs']}</div>
                </div>
                <div>
                    <div style="font-size:0.65rem; color:#8892a4;
                                text-transform:uppercase;">Returns on {stake:.0f}u</div>
                    <div style="font-size:1.3rem; font-weight:700;
                                color:#f5d020;">{potential_return(selected['total_odds'], stake):.2f}u</div>
                </div>
            </div>

            <div style="margin-top:16px; padding:10px 14px; background:#0a1220;
                        border-radius:4px; border-left:2px solid #7c3aed;">
                <span style="font-size:0.7rem; color:#8892a4;">
                    ⚠️ For research purposes only. Gamble responsibly.
                    This is a probability model, not a guarantee.
                </span>
            </div>
        </div>
        """
        render_html(panel_html)

        # Clear button
        render_html("<div style='margin-top:12px;'></div>")
        if st.button("✕  Clear Selection"):
            st.session_state.selected_slip_id = None
            st.rerun()


# ============================================================
#  FOOTER
# ============================================================
render_html("""
<div style="margin-top:48px; padding-top:16px; border-top:1px solid #1e2d45;
            text-align:center; font-size:0.65rem; color:#4a5568;
            text-transform:uppercase; letter-spacing:0.12em;">
    BTTS Slip AI Dashboard · Local Mode · Heuristic Model v1.0 ·
    Upgrade path: XGBoost + API-Football
</div>
""")
