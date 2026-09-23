"""
app.py
------
Football Match Outlook — an information dashboard.

Run with:
    streamlit run app.py

What this is: today's fixtures, each side's recent form from real results, and
a model's probability for the match. That is all it claims to be.

What it is NOT, and used to be: a betting tool. It was built as one, and then
the models were tested properly — walk-forward, against closing bookmaker odds,
on 17,697 out-of-sample matches. They lost on every outcome, and the home-win
loss was statistically significant (ROI -4.6%, t = -2.91). The betting
apparatus was removed rather than left in place looking authoritative.

The evidence is still in the repo (models/market_test.py, models/bakeoff.py)
because a negative result is worth keeping. See the README.

Layout:
  [Sidebar]    Source, market, leagues, matchday
  [Section 1]  The model's strongest calls
  [Section 2]  Every fixture, with form
  [Panel]      International pricing (manual pairing)
"""

import sys
from pathlib import Path

# Ensure local modules are importable regardless of how/where Streamlit launches.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from data.sample_data       import load_matches
from data.live_fixtures     import (DEFAULT_LEAGUES, LEAGUES as LIVE_LEAGUES,
                                    available_dates, build_fixture_card,
                                    feed_status, load_leagues,
                                    malformed_score_count, next_matchday)
from data.international     import (DEFAULT_SINCE, fit_international_model,
                                    load_results, price_fixture, rateable_teams,
                                    recent_form, team_match_counts)
from models.btts_model      import score_matches
from models.match_result    import build_match_result_card
from models.daily_picks     import confidence_band, rank_picks


SEASONS = ["2026-27", "2025-26"]


# ============================================================
#  PAGE CONFIG  (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title   = "Football Match Outlook",
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
render_html("""<style>
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
</style>""")


# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    render_html("""
    <div style="text-align:center; padding: 16px 0 24px 0;">
        <div style="font-size:2rem;">⚽</div>
        <div style="font-size:0.7rem; color:#00e5ff; letter-spacing:0.2em;
                    text-transform:uppercase; font-weight:700;">Match Outlook</div>
        <div style="font-size:0.6rem; color:#8892a4; margin-top:4px;">
            Fixtures · Form · Probabilities
        </div>
    </div>
    """)

    st.markdown("---")

    st.markdown("**📡 Data Source**")
    source = st.radio(
        "Source",
        ["Live fixtures", "Sample data", "International (manual)"],
        index=0,
        help="Live: real fixtures and real form from the openfootball feed. "
             "Sample: a synthetic demo card. International: price any two "
             "national teams — no fixture list exists for internationals.",
        label_visibility="collapsed",
    )
    live_mode          = source == "Live fixtures"
    international_mode = source.startswith("International")

    market_choice = st.radio(
        "Market",
        ["Match Result (1X2)", "Both Teams To Score"],
        index=0,
        help="Match result is the market the backtest found signal in. "
             "BTTS scored no better than the league base rate.",
        label_visibility="collapsed",
    )
    result_mode = market_choice.startswith("Match Result")
    prob_label  = "Win Prob" if result_mode else "BTTS Prob"

    if result_mode and not live_mode and not international_mode:
        st.caption("⚠️ Match result needs real results to fit on — "
                   "switching the source to live fixtures.")
        live_mode = True

    if live_mode:
        live_league_codes = st.multiselect(
            "Leagues",
            options=list(LIVE_LEAGUES),
            default=DEFAULT_LEAGUES,
            format_func=lambda code: LIVE_LEAGUES[code],
        )

    if not international_mode:
        st.markdown(f"**🎯 Show fixtures above**")
        min_prob = st.slider(
            f"Min {prob_label} (%)",
            min_value=0, max_value=90,
            value=0, step=5,
            help="A display filter only — the strongest calls above are always "
                 "drawn from the whole card.",
        ) / 100.0

    st.markdown("---")
    if st.button("🔄  Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ============================================================
#  DATA
# ============================================================
@st.cache_data(ttl=300)
def get_sample_card() -> pd.DataFrame:
    """The synthetic demo card, scored."""
    return score_matches(load_matches())


@st.cache_data(ttl=1800)
def get_live_season(codes: tuple[str, ...]) -> pd.DataFrame:
    """
    Every fixture and result for the chosen leagues, this season and last.

    Last season is included so form is not built from three August matches,
    and it is cached for half an hour — the feed updates daily.
    """
    return load_leagues(SEASONS, codes=list(codes))


@st.cache_data(ttl=86400, show_spinner="Loading international results …")
def get_international_results(since: str) -> pd.DataFrame:
    return load_results(since=since)


@st.cache_resource(show_spinner="Fitting national-team ratings …")
def get_international_model(since: str):
    return fit_international_model(load_results(since=since))


# ============================================================
#  INTERNATIONAL — MANUAL PAIRING
# ============================================================
# National-team football has no fixture feed reachable from here, so there is
# no card to show. You name the two teams instead.
if international_mode:
    results = get_international_results(DEFAULT_SINCE)
    teams   = rateable_teams(results)
    counts  = team_match_counts(results)

    render_html(f"""
    <div style="padding: 8px 0 4px 0;">
        <span style="font-size:1.6rem; font-weight:800; color:#e8eaf0;
                     letter-spacing:0.05em;">INTERNATIONAL</span>
        <span style="font-size:1.6rem; font-weight:800; color:#00e5ff;
                     letter-spacing:0.05em;"> MATCH OUTLOOK</span>
        <div style="font-size:0.7rem; color:#8892a4; margin-top:2px;
                    text-transform:uppercase; letter-spacing:0.15em;">
            Manual Pairing · {len(teams)} National Teams ·
            {len(results):,} Results Since {DEFAULT_SINCE[:4]}
        </div>
    </div>
    """)

    render_html("""
    <div style="background:#0d1525; border:1px solid #1e2d45; border-left:2px solid #f5d020;
                border-radius:6px; padding:12px 16px; margin:12px 0;
                font-size:0.72rem; color:#8892a4; line-height:1.6;">
        <strong style="color:#f5d020;">Why manual?</strong> No source reachable from here
        lists upcoming international fixtures — the results archive is history only.
        <br>
        <strong style="color:#ff9d00;">Held to a lower standard than the club pages.</strong>
        The models were validated on club leagues; nothing here has been backtested on
        international football. National sides play a handful of matches a year and
        friendlies are played with reserves, so treat this as a considered estimate.
    </div>
    """)

    pick_home, pick_away, pick_venue = st.columns([2, 2, 1])
    with pick_home:
        home_team = st.selectbox("Home / first team", teams,
                                 index=teams.index("Uzbekistan") if "Uzbekistan" in teams else 0)
    with pick_away:
        others = [t for t in teams if t != home_team]
        away_team = st.selectbox("Away / second team", others,
                                 index=others.index("Iran") if "Iran" in others else 0)
    with pick_venue:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        neutral = st.checkbox("Neutral venue", value=True,
                              help="On neutral ground the home-advantage term is dropped.")

    priced = price_fixture(get_international_model(DEFAULT_SINCE),
                           home_team, away_team, neutral=neutral)

    thin = [t for t in (home_team, away_team) if counts.get(t, 0) < 10]
    if thin:
        render_html(f"""
        <div style="background:#1a1400; border:1px solid #f5d02044; border-radius:6px;
                    padding:10px 14px; margin:8px 0; color:#f5d020; font-size:0.75rem;">
            ⚠️ Thin data: {", ".join(f"{t} ({int(counts.get(t, 0))} matches)" for t in thin)}
            since {DEFAULT_SINCE[:4]}. Ratings built on this little are unreliable.
        </div>
        """)

    outcomes = [
        (f"{home_team} win", priced["p_home"], "#00ff88"),
        ("Draw",             priced["p_draw"], "#f5d020"),
        (f"{away_team} win", priced["p_away"], "#00e5ff"),
    ]
    cards = "".join(f"""
        <div class="metric-card" style="border-top-color:{colour};">
            <div style="font-size:0.65rem; color:#8892a4; text-transform:uppercase;
                        letter-spacing:0.08em;">{name}</div>
            <div style="font-size:1.8rem; font-weight:800; color:{colour};
                        line-height:1.2;">{prob*100:.1f}%</div>
        </div>
    """ for name, prob, colour in outcomes)

    render_html(f"""
    <div class="metric-row" style="margin-top:16px;">{cards}</div>
    <div class="metric-row">
        <div class="metric-card">
            <div style="font-size:0.65rem; color:#8892a4; text-transform:uppercase;">Expected goals</div>
            <div style="font-size:1.4rem; font-weight:700; color:#e8eaf0;">
                {priced['xg_home']} – {priced['xg_away']}</div>
        </div>
        <div class="metric-card">
            <div style="font-size:0.65rem; color:#8892a4; text-transform:uppercase;">Both teams to score</div>
            <div style="font-size:1.4rem; font-weight:700; color:#e8eaf0;">
                {priced['p_btts']*100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div style="font-size:0.65rem; color:#8892a4; text-transform:uppercase;">Over 2.5 goals</div>
            <div style="font-size:1.4rem; font-weight:700; color:#e8eaf0;">
                {priced['p_over25']*100:.1f}%</div>
        </div>
        <div class="metric-card">
            <div style="font-size:0.65rem; color:#8892a4; text-transform:uppercase;">Venue</div>
            <div style="font-size:1.4rem; font-weight:700; color:#e8eaf0;">
                {"Neutral" if neutral else home_team}</div>
        </div>
    </div>
    """)

    render_html('<div class="section-header"><h3>Recent Form</h3></div>')
    form_left, form_right = st.columns(2)
    for column, team in ((form_left, home_team), (form_right, away_team)):
        with column:
            st.markdown(f"**{team}** · last 8")
            form = recent_form(results, team, window=8)
            if form.empty:
                st.caption("No matches in the window.")
            else:
                st.dataframe(form, hide_index=True, use_container_width=True)

    render_html("""
    <div style="margin-top:16px; padding:10px 14px; background:#0a1220;
                border-radius:4px; border-left:2px solid #7c3aed;">
        <span style="font-size:0.7rem; color:#8892a4;">
            Probabilities from a statistical model, for interest and context.
            Not betting advice — see the README for how these models performed
            against bookmaker prices.
        </span>
    </div>
    """)
    st.stop()


# ============================================================
#  BUILD THE CARD
# ============================================================
live_meta: dict = {}

if live_mode:
    season_df = get_live_season(tuple(live_league_codes))

    if season_df.empty:
        st.error("The live fixture feed returned nothing — it may be unreachable "
                 "from here. Switch to Sample data in the sidebar to keep working.")
        st.stop()

    upcoming    = available_dates(season_df)
    default_day = next_matchday(season_df)

    with st.sidebar:
        st.markdown("**📅 Matchday**")
        if upcoming:
            chosen_day = st.selectbox(
                "Date", options=upcoming,
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

    if result_mode:
        priced, result_meta = build_match_result_card(card_df, season_df)
        priced = priced[priced["selection_prob"].notna()].reset_index(drop=True)
        if priced.empty:
            render_html("""
            <div style="background:#1a1400; border:1px solid #f5d02044; border-radius:6px;
                        padding:16px; color:#f5d020; font-size:0.85rem;">
                No league on this card has enough completed matches to fit a model yet.
                Pick a later matchday, or switch the market to Both Teams To Score.
            </div>
            """)
            st.stop()
        # One column carries "the probability being shown", whichever market it is.
        priced["prob"] = priced["selection_prob"]
        card = priced
    else:
        card = score_matches(card_df)
        card["prob"] = card["btts_prob"]
        result_meta = {}

    live_meta = {
        "day":      chosen_day,
        "leagues":  len(live_league_codes),
        "results":  int(season_df["played"].sum()),
        "unusable": malformed_score_count(season_df),
        "fetched":  feed_status(SEASONS, live_league_codes).get("last_fetched"),
        **result_meta,
    }
else:
    card = get_sample_card()
    card["prob"] = card["btts_prob"]
    result_meta = {}


# ============================================================
#  PAGE HEADER
# ============================================================
col_title, col_status = st.columns([3, 1])
with col_title:
    render_html(f"""
    <div style="padding: 8px 0 4px 0;">
        <span style="font-size:1.6rem; font-weight:800; color:#e8eaf0;
                     letter-spacing:0.05em;">MATCH</span>
        <span style="font-size:1.6rem; font-weight:800; color:#00e5ff;
                     letter-spacing:0.05em;"> OUTLOOK</span>
        <div style="font-size:0.7rem; color:#8892a4; margin-top:2px;
                    text-transform:uppercase; letter-spacing:0.15em;">
            {"Match Result · 1X2" if result_mode else "Both Teams To Score"} ·
            Fixtures · Form · Probabilities
        </div>
    </div>
    """)
with col_status:
    if live_mode:
        chip_border, chip_bg = "#00ff88", "#0d2518"
        chip_label = f"● LIVE FEED · {live_meta['day']:%a %d %b} · {len(card)} Fixtures"
    else:
        chip_border, chip_bg = "#f5d020", "#1a1400"
        chip_label = f"◌ SAMPLE DATA · {len(card)} Fixtures"

    render_html(f"""
    <div style="text-align:right; padding-top:12px;">
        <span style="background:{chip_bg}; border:1px solid {chip_border}; color:{chip_border};
                     padding:4px 10px; border-radius:20px; font-size:0.68rem;
                     letter-spacing:0.1em; text-transform:uppercase;">{chip_label}</span>
    </div>
    """)


# ============================================================
#  PROVENANCE + WHAT THIS IS
# ============================================================
if live_mode:
    fetched = live_meta.get("fetched")
    fitted  = (f" · Poisson models fitted per league on "
               f"{live_meta['training_matches']:,} results"
               if result_mode and "training_matches" in live_meta else "")
    render_html(f"""
    <div style="background:#0d1525; border:1px solid #1e2d45; border-left:2px solid #00e5ff;
                border-radius:6px; padding:12px 16px; margin:8px 0 4px 0;
                font-size:0.72rem; color:#8892a4; line-height:1.6;">
        <strong style="color:#00e5ff;">Real fixtures.</strong>
        {live_meta['leagues']} leagues · form built from
        <strong style="color:#e8eaf0;">{live_meta['results']:,}</strong> completed matches
        (this season and last){f" · feed fetched {fetched:%d %b %H:%M}" if fetched is not None else ""}
        {f" · {live_meta['unusable']} matches skipped for unusable scores" if live_meta['unusable'] else ""}{fitted}.
        <br>
        <strong style="color:#f5d020;">These are descriptive probabilities, not betting
        advice.</strong> Tested walk-forward against closing bookmaker odds on 17,697
        matches, this model lost on every outcome — ROI −4.6% on home wins, and that
        result was statistically significant. It reads matches; it does not beat markets.
    </div>
    """)


# ============================================================
#  SECTION 1: THE MODEL'S STRONGEST CALLS
# ============================================================
top_picks = rank_picks(card, top_n=5, prob_column="prob")

render_html(f"""
<div class="section-header"><h3>01 · Strongest Calls{
    f" · {live_meta['day']:%A %d %B}" if live_mode else ""}</h3></div>
""")

if top_picks.empty:
    render_html("""
    <div style="background:#1a1400; border:1px solid #f5d02044; border-radius:6px;
                padding:14px; color:#f5d020; font-size:0.8rem;">
        Nothing on this card could be priced.
    </div>
    """)
else:
    calls = ""
    for rank, pick in enumerate(top_picks.itertuples(index=False), 1):
        band, colour = confidence_band(pick.prob)
        headline = (getattr(pick, "selection_label", None)
                    if result_mode else f"Both teams to score")
        calls += f"""
        <div style="display:flex; align-items:center; gap:14px; padding:10px 14px;
                    background:#111827; border:1px solid #1e2d45;
                    border-left:3px solid {colour}; border-radius:6px; margin-bottom:8px;">
            <div style="font-size:1.1rem; font-weight:800; color:{colour};
                        min-width:26px;">{rank}</div>
            <div style="flex:1;">
                <div style="font-size:0.9rem; font-weight:700; color:#e8eaf0;">{headline}</div>
                <div style="font-size:0.68rem; color:#8892a4;">
                    {pick.home_team} vs {pick.away_team} · {pick.league} · {pick.kickoff}
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:1.2rem; font-weight:800; color:{colour};">
                    {pick.prob*100:.1f}%</div>
                <div style="font-size:0.66rem; color:#8892a4;">{band}</div>
            </div>
        </div>
        """
    render_html(calls)
    render_html("""
    <div style="font-size:0.68rem; color:#8892a4; margin-top:4px;">
        Ranked by the model's confidence — which says how predictable a match looks,
        not whether any price on it is worth taking.
    </div>
    """)


# ============================================================
#  SECTION 2: EVERY FIXTURE
# ============================================================
render_html('<div class="section-header"><h3>02 · All Fixtures</h3></div>')

shown = card[card["prob"] >= min_prob] if min_prob > 0 else card

if shown.empty:
    render_html(f"""
    <div style="background:#1a0a0a; border:1px solid #ff444444; border-radius:6px;
                padding:16px; color:#ff7777; font-size:0.82rem; text-align:center;">
        No fixture on this card reaches {min_prob*100:.0f}%. Lower the filter in the sidebar.
    </div>
    """)
else:
    if min_prob > 0:
        render_html(f"""
        <div style='font-size:0.75rem; color:#8892a4; margin-bottom:8px;'>
            Showing <strong style='color:#00e5ff;'>{len(shown)}</strong> of {len(card)}
            fixtures · {prob_label} ≥ {min_prob*100:.0f}%
        </div>
        """)

    rows_html = ""
    for _, row in shown.sort_values("prob", ascending=False).iterrows():
        band, colour = confidence_band(row["prob"])
        selection_cell = ""
        if result_mode:
            selection_cell = (f'<td style="color:#00e5ff; font-size:0.78rem; '
                              f'font-weight:700;">{row.get("selection_label") or "—"}</td>')
        rows_html += f"""
        <tr>
            <td style="color:#8892a4; font-size:0.7rem;">{row['kickoff']}</td>
            <td><strong>{row['home_team']}</strong>
                <span style="color:#8892a4; font-size:0.72rem;"> vs </span>
                <strong>{row['away_team']}</strong></td>
            <td style="color:#8892a4;">{row['league']}</td>
            {selection_cell}
            <td>
                <span class="prob-badge" style="background:{colour}22; color:{colour};
                      border:1px solid {colour}44;">{row['prob']*100:.1f}%</span>
            </td>
            <td style="font-size:0.75rem; color:#8892a4;">{band}</td>
            <td style="color:#8892a4; font-size:0.75rem;">
                {row['home_avg_scored']} / {row['away_avg_scored']}
            </td>
            <td style="color:#8892a4; font-size:0.75rem;">
                {row['home_avg_conceded']} / {row['away_avg_conceded']}
            </td>
        </tr>
        """

    render_html(f"""
    <table class="match-table">
        <thead>
            <tr>
                <th>Kickoff</th>
                <th>Match</th>
                <th>League</th>
                {'<th>Most likely</th>' if result_mode else ''}
                <th>{prob_label}</th>
                <th>Confidence</th>
                <th>Avg Scored H/A</th>
                <th>Avg Conceded H/A</th>
            </tr>
        </thead>
        <tbody>{rows_html}</tbody>
    </table>
    """)


# ============================================================
#  FOOTER
# ============================================================
render_html("""
<div style="margin-top:48px; padding-top:16px; border-top:1px solid #1e2d45;
            text-align:center; font-size:0.65rem; color:#4a5568;
            text-transform:uppercase; letter-spacing:0.12em;">
    Match Outlook · Information only · Model loses to closing odds — see README
</div>
""")
