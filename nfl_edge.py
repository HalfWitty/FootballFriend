"""
NFL Edge Board  (FootballFriend)
================================

Builds edge.html - the NFL version of the NHL Edge Board - from:
  * the FootballFriend data/ folder (Sleeper weekly stats, incl. TEAM_ rows)
  * nflverse's public games file (scores, kickoff times, closing spreads,
    totals, moneylines and their prices)

For the target week it projects every game, prices moneyline / spread /
total / team totals, projects player props, and picks the top bets. It then
re-runs the same model for every earlier week of the season - each time using
ONLY data from before that week - and grades those predictions against what
actually happened (the Track record tab).

Usage:
    pip install requests pandas numpy
    python nfl_edge.py              # run from the FootballFriend repo folder

Each week: run pull_2026_stats.py first (fresh stats), then this script.
"""

import base64
import hashlib
import html
import io
import math
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SEASON = 2026
TARGET_WEEK = None        # None = the first week with unplayed games. Set a number to rebuild a past week.
FIRST_GRADED_WEEK = 2     # track record covers weeks FIRST_GRADED_WEEK..target (walk-forward)

DATA_DIR = "data"         # local FootballFriend data folder; falls back to GitHub if missing
REPO_RAW = "https://raw.githubusercontent.com/HalfWitty/FootballFriend/main/data"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
OUTPUT_HTML = "edge.html"
LOG_CSV = "nfl_data/predictions_log.csv"
SITE_LINK = "index.html"  # link back to The Stat Sheet

# --- team model (tuned on the 2025 season only, never on 2026) ---
HALF_LIFE_WEEKS = 6       # a game's weight halves every N weeks
LAST_SEASON_WEIGHT = 0.35 # extra discount per season back
SEASONS_BACK = 2
RIDGE = 4.0               # = this many games of league-average play mixed into every rating
POINTS_BLEND = 0.7        # rating target: 70% actual points, 30% yardage-based expected points
MARGIN_SCALE = 1.0        # multiplier on the projected margin (calibration)
# How much of the model's disagreement with the market line to price in (1 = pure model, 0 = market only).
# Fitted on 2023-2025 the best value was ~0 for sides and ~0.1 for totals, i.e. the model hasn't beaten closing
# lines. Left at 1.0 so the board shows the model's own opinion; lower it to anchor prices to the market.
MODEL_WEIGHT_SIDES = 1.0
MODEL_WEIGHT_TOTALS = 1.0

# --- betting rules (same idea as the NHL board) ---
HIGH_PROB = 0.58          # "High probability" tag
MIN_PROB = 0.30           # no longshots
MAX_MARKET_GAP = 0.08     # skip if model is this far from the market's no-vig % (usually news)
CUSHION_SIDES = 0.035     # Bet-at cushion for ML / spread / total
CUSHION_EARLY = 0.015     # extra cushion through week 6
CUSHION_PROPS = 0.06
MAX_BETS_PER_GAME = 2
STAKE = 1.0               # units per graded bet
LONG_RUN_SEASONS = [2023, 2024, 2025]   # full past seasons re-run under the same rules (Track record)

# --- GitHub auto-upload (same token setup as pull_2026_stats.py) ---
PUSH_TO_GITHUB = False
GITHUB_REPO = "HalfWitty/FootballFriend"
GITHUB_BRANCH = "main"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
# ------------------------------------------------------------------

SLEEPER_TO_NFLVERSE = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "STL": "LA"}
TEAM_NAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills", "CAR": "Panthers",
    "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns", "DAL": "Cowboys", "DEN": "Broncos",
    "DET": "Lions", "GB": "Packers", "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars",
    "KC": "Chiefs", "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants", "NYJ": "Jets",
    "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks", "SF": "49ers", "TB": "Buccaneers",
    "TEN": "Titans", "WAS": "Commanders"}
DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SEA", "SF"]}
TEAM_DIV = {t: d for d, ts in DIVISIONS.items() for t in ts}

STAT_COLS = ["pass_yd", "pass_att", "pass_cmp", "pass_td", "pass_int", "rush_yd", "rush_att",
             "rush_td", "rec", "rec_yd", "rec_td", "rec_tgt", "off_snp", "fum_lost", "off_yd",
             "fd", "pass_sack", "gp"]


# ==================================================================
# DATA
# ==================================================================
def read_csv_any(name, **kw):
    local = os.path.join(DATA_DIR, name)
    if os.path.exists(local):
        return pd.read_csv(local, low_memory=False, **kw)
    r = requests.get(f"{REPO_RAW}/{name}", timeout=120)
    if r.status_code != 200:
        return None
    return pd.read_csv(io.StringIO(r.text), low_memory=False, **kw)


def stat_files(season):
    if season >= 2026:
        return [f"stats_{season}_part1.csv", f"stats_{season}_part2.csv"]
    if season >= 2024:
        return [f"stats_{season}_weekly_part1.csv", f"stats_{season}_weekly_part2.csv"]
    return [f"stats_{season}_part1.csv", f"stats_{season}_part2.csv"]


def load_stats(seasons):
    """Weekly Sleeper rows for the given seasons, team codes in nflverse style."""
    players = read_csv_any("players_nfl.csv")
    pos_map = dict(zip(players.player_id.astype(str), players.position)) if players is not None else {}
    name_map = dict(zip(players.player_id.astype(str), players.full_name)) if players is not None else {}
    frames = []
    keep = ["player_id", "full_name", "position", "team", "opponent", "season", "week"] + STAT_COLS
    for s in seasons:
        header = None
        for f in stat_files(s):
            df = read_csv_any(f)
            if df is None:
                print(f"  [warn] missing {f}")
                continue
            if "season" not in df.columns and header is not None and len(header) == df.shape[1]:
                # a part2 file saved without its header row: reuse part1's
                first = pd.DataFrame([list(df.columns)], columns=header)
                df.columns = header
                df = pd.concat([first, df], ignore_index=True)
            header = list(df.columns)
            if "team" not in df.columns and "team_x" in df.columns:
                df = df.rename(columns={"team_x": "team"})
            df = df.reindex(columns=keep)
            frames.append(df)
    st = pd.concat(frames, ignore_index=True)
    st["player_id"] = st.player_id.astype(str)
    st = st.drop_duplicates(["player_id", "season", "week"])
    st["position"] = st.position.fillna(st.player_id.map(pos_map))
    st["full_name"] = st.full_name.fillna(st.player_id.map(name_map))
    for c in ("team", "opponent"):
        st[c] = st[c].replace(SLEEPER_TO_NFLVERSE)
    st[STAT_COLS] = st[STAT_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    st["season"] = pd.to_numeric(st.season, errors="coerce")
    st["week"] = pd.to_numeric(st.week, errors="coerce")
    st = st.dropna(subset=["season", "week"])
    st["season"] = st.season.astype(int)
    st["week"] = st.week.astype(int)
    return st


def load_games():
    r = requests.get(GAMES_URL, timeout=120)
    r.raise_for_status()
    g = pd.read_csv(io.StringIO(r.text), low_memory=False)
    g = g[g.season >= 2006].copy()
    g["done"] = g.home_score.notna() & g.away_score.notna()
    g["ord"] = g.season * 25 + g.week
    return g


def team_game_table(games, stats):
    """One row per team per finished game: points for/against + yardage-based expected points."""
    done = games[games.done]
    home = pd.DataFrame({"season": done.season, "week": done.week, "ord": done.ord, "game_id": done.game_id,
                         "team": done.home_team, "opp": done.away_team, "pf": done.home_score,
                         "pa": done.away_score, "home": (done.location != "Neutral").astype(int)})
    away = pd.DataFrame({"season": done.season, "week": done.week, "ord": done.ord, "game_id": done.game_id,
                         "team": done.away_team, "opp": done.home_team, "pf": done.away_score,
                         "pa": done.home_score, "home": 0})
    tg = pd.concat([home, away], ignore_index=True)
    team_rows = stats[stats.player_id.str.startswith("TEAM_")].copy()
    team_rows["team"] = team_rows.player_id.str[5:].replace(SLEEPER_TO_NFLVERSE)
    team_rows["to"] = team_rows.pass_int + team_rows.fum_lost
    tr = team_rows[["season", "week", "team", "off_yd", "fd", "to", "pass_att", "rush_att", "pass_sack"]]
    tg = tg.merge(tr, on=["season", "week", "team"], how="left")
    tg["plays"] = tg.pass_att + tg.rush_att + tg.pass_sack
    tg["has_box"] = tg.off_yd.notna() & (tg.off_yd > 0)
    return tg


def fit_xpts(tg):
    """Points ~ yards + first downs + turnovers (OLS). Steadier than raw points."""
    d = tg[tg.has_box]
    X = np.column_stack([np.ones(len(d)), d.off_yd, d.fd, d.to])
    coef, *_ = np.linalg.lstsq(X, d.pf.values, rcond=None)
    return coef


def add_xpts(tg, coef):
    tg = tg.copy()
    x = coef[0] + coef[1] * tg.off_yd + coef[2] * tg.fd + coef[3] * tg.to
    tg["xpf"] = np.where(tg.has_box, x, tg.pf)
    opp = tg[["game_id", "team", "xpf"]].rename(columns={"team": "opp", "xpf": "xpa"})
    return tg.merge(opp, on=["game_id", "opp"], how="left")


# ==================================================================
# TEAM MODEL
# ==================================================================
def fit_ratings(tg, season, week, params=None):
    """Recency-weighted ridge: score = mu + Off[team] + Def[opp] + HFA*home."""
    p = dict(hl=HALF_LIFE_WEEKS, lsw=LAST_SEASON_WEIGHT, ridge=RIDGE, blend=POINTS_BLEND)
    p.update(params or {})
    ord_t = season * 25 + week
    d = tg[(tg.ord < ord_t) & (tg.season >= season - SEASONS_BACK)].copy()
    teams = sorted(set(d.team) | set(d.opp))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    # weeks ago, counting only game weeks (the offseason is handled by LAST_SEASON_WEIGHT)
    sb = (season - d.season).values
    age = np.where(sb == 0, week - d.week.values,
                   week + np.maximum(0, 18 - d.week.values) + 18 * np.maximum(sb - 1, 0))
    w =0.5 ** (age / p["hl"]) * p["lsw"] ** (season - d.season.values)
    y = p["blend"] * d.pf.values + (1 - p["blend"]) * d.xpf.values
    X = np.zeros((len(d), 2 * n + 2))
    X[:, 0] = 1
    X[np.arange(len(d)), 1 + d.team.map(idx).values] = 1
    X[np.arange(len(d)), 1 + n + d.opp.map(idx).values] = 1
    X[:, -1] = d.home.values
    W = w[:, None]
    A = X.T @ (X * W)
    reg = np.full(2 * n + 2, p["ridge"])
    reg[0] = 1e-6
    reg[-1] = 1e-6
    A += np.diag(reg)
    b = np.linalg.solve(A, X.T @ (w * y))
    return {"mu": b[0], "hfa": b[-1], "off": dict(zip(teams, b[1:1 + n])),
            "def": dict(zip(teams, b[1 + n:1 + 2 * n])), "teams": teams}


def project_game(r, home, away, neutral=False):
    h = 0 if neutral else r["hfa"]
    hp = r["mu"] + r["off"].get(home, 0) + r["def"].get(away, 0) + h
    ap = r["mu"] + r["off"].get(away, 0) + r["def"].get(home, 0)
    return hp, ap


class MarginDist:
    """P(home margin = k) built from real NFL results near the same spread (keeps 3/7 key numbers)."""

    def __init__(self, games, before_season):
        h = games[(games.season < before_season) & (games.season >= 2010) & games.done &
                  games.spread_line.notna()]
        self.spread = h.spread_line.values.astype(float)
        self.result = h.result.values.astype(int)
        tot = h[h.total_line.notna()]
        self.total_sd = float(np.std(tot.total - tot.total_line))
        self.cache = {}

    def pmf(self, m):
        key = round(m * 4) / 4
        if key in self.cache:
            return self.cache[key]
        bw = 1.25
        while True:
            w = np.exp(-0.5 * ((self.spread - key) / bw) ** 2)
            if w.sum() >= 250 or bw > 6:
                break
            bw *= 1.4
        res = self.result + np.round(key - np.average(self.spread, weights=w)).astype(int)
        ks = np.arange(-70, 71)
        pm = np.bincount(res + 70, weights=w, minlength=141)[:141]
        pm = pm / pm.sum()
        self.cache[key] = (ks, pm)
        return ks, pm

    def cover(self, m, line_home):
        """Home covers line_home (e.g. -3 means home gives 3). Returns (win, push, loss)."""
        ks, pm = self.pmf(m)
        v = ks + line_home
        return pm[v > 0].sum(), pm[v == 0].sum(), pm[v < 0].sum()


def total_probs(mu, line, sd):
    """Over/push/under for a game total with a discretised normal."""
    ks = np.arange(0, 121)
    cdf = lambda x: 0.5 * (1 + math.erf((x - mu) / (sd * math.sqrt(2))))
    pm = np.array([cdf(k + 0.5) - cdf(k - 0.5) for k in ks])
    pm /= pm.sum()
    return pm[ks > line].sum(), pm[ks == line].sum(), pm[ks < line].sum()


# ==================================================================
# ODDS HELPERS
# ==================================================================
def am_to_dec(a):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return None
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def prob_to_am(p):
    p = min(max(p, 0.001), 0.999)
    return -round(100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def fmt_am(a):
    if a is None or (isinstance(a, float) and math.isnan(a)):
        return "-"
    a = int(round(a))
    return f"+{a}" if a > 0 else str(a)


def no_vig(a1, a2):
    d1, d2 = am_to_dec(a1), am_to_dec(a2)
    if not d1 or not d2:
        return None
    i1, i2 = 1 / d1, 1 / d2
    return i1 / (i1 + i2)


# ==================================================================
# PLAYER PROPS
# ==================================================================
PROP_STATS = {   # stat: (label, positions, kind, milestones, game-script exponent)
    "pass_yd": ("Pass yds", ("QB",), "n", (200, 225, 250, 275, 300), 0.25),
    "pass_td": ("Pass TDs", ("QB",), "p", (2, 3), 0.6),
    "rush_yd": ("Rush yds", ("QB", "RB"), "n", (40, 60, 80, 100), 0.5),
    "rec": ("Receptions", ("RB", "WR", "TE"), "p", (3, 4, 5, 6, 7), 0.15),
    "rec_yd": ("Rec yds", ("RB", "WR", "TE"), "n", (40, 60, 80, 100), 0.25),
    "td": ("Anytime TD", ("QB", "RB", "WR", "TE"), "p", (1,), 0.9),
}
PLAYER_HALF_LIFE = 5      # games played
MIN_PROP_LINE = {"pass_yd": 170, "pass_td": 1.0, "rush_yd": 30, "rec": 2.5, "rec_yd": 30, "td": 0.25}
PLAYER_LAST_SEASON = 0.5
OPP_SHRINK = 5            # games of league-average defense mixed into the opponent factor
OPP_WEIGHT = 0.6          # how much of the (shrunk) opponent factor to apply
# residual sd for yardage props: sd = a + b * projection (fit on 2024-2025)
YARD_SD = {"pass_yd": (65.0, 0.0), "rush_yd": (15.0, 0.36), "rec_yd": (16.0, 0.34)}
# anytime-TD rates are noisy; calibrated lambda = a + b * raw (fit on 2025)
TD_CALIB = (0.12, 0.5)
PROP_CALIB_K = 0.85       # yardage/reception probabilities were a bit overconfident in 2025; shrink toward 50%
TEAM_SCORE_SD = 9.2


def norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def pois_sf(lam, k):
    """P(X >= k) for Poisson(lam)."""
    if k <= 0:
        return 1.0
    s, term = 0.0, math.exp(-lam)
    for i in range(k):
        s += term
        term *= lam / (i + 1)
    return max(0.0, 1 - s)


def prob_at_least(kind, mean, sd, k):
    if kind == "n":
        return 1 - norm_cdf((k - 0.5 - mean) / sd)
    return pois_sf(mean, int(k))


def prop_sd(stat, mean):
    a, b = YARD_SD.get(stat, (0, 0))
    return a + b * mean


def player_pool(st, season, week):
    d = st[(st.season >= season - 1) & ((st.season < season) | (st.week < week)) &
           st.position.isin(["QB", "RB", "WR", "TE"]) & (st.off_snp > 0)].copy()
    d["td"] = d.rush_td + d.rec_td
    d = d.sort_values(["season", "week"])
    return d


def defense_factors(st, season, week):
    """How much each defense allows to each position group vs league average (recency weighted, shrunk)."""
    d = player_pool(st, season, week)
    sb = season - d.season.values
    age = np.where(sb == 0, week - d.week.values, week + np.maximum(0, 18 - d.week.values))
    d["w"] = 0.5 ** (age / HALF_LIFE_WEEKS) * LAST_SEASON_WEIGHT ** sb
    out = {}
    for stat, (_, poss, *_r) in PROP_STATS.items():
        for pos in poss:
            g = d[d.position == pos].groupby(["opponent", "season", "week"]).agg(v=(stat, "sum"), w=("w", "first"))
            g = g.reset_index()
            lg = np.average(g.v, weights=g.w) if len(g) else 0
            if lg <= 0:
                continue
            for opp, x in g.groupby("opponent"):
                f = (np.sum(x.w * x.v) + OPP_SHRINK * lg) / (x.w.sum() + OPP_SHRINK) / lg
                out[(opp, stat, pos)] = 1 + OPP_WEIGHT * (f - 1)
    return out


def project_players(st, season, week, game_rows, team_pts, team_avg_pts, def_f):
    """Projections for every likely-active skill player in this week's games."""
    pool = player_pool(st, season, week)
    rows = []
    for _, gm in game_rows.iterrows():
        for team, opp, qb_name in ((gm.home_team, gm.away_team, gm.get("home_qb_name")),
                                   (gm.away_team, gm.home_team, gm.get("away_qb_name"))):
            tp = pool[pool.team == team]
            if tp.empty:
                continue
            recent = tp[tp.season == tp.season.max()]
            last_weeks = sorted(recent.week.unique())[-2:]
            active = recent[recent.week.isin(last_weeks)]
            ratio = team_pts[team] / max(team_avg_pts.get(team, team_pts[team]), 10)
            for pid, pr in active.groupby("player_id"):
                hist = pool[pool.player_id == pid]
                hist = hist[hist.team == team] if hist[hist.team == team].shape[0] >= 2 else hist
                pos = hist.position.iloc[-1]
                name = hist.full_name.iloc[-1]
                if pos == "QB":
                    if qb_name and isinstance(qb_name, str):
                        if name != qb_name:
                            continue
                    elif pr.pass_att.max() < 10:
                        continue
                n = len(hist)
                ago = np.arange(n)[::-1]
                w = 0.5 ** (ago / PLAYER_HALF_LIFE) * np.where(hist.season.values < season, PLAYER_LAST_SEASON, 1)
                touches = np.average(hist.rush_att + hist.rec_tgt, weights=w)
                if pos != "QB" and touches < 3:
                    continue
                cur = hist[hist.season == season]
                naive_src = cur if len(cur) else hist
                info = {"player_id": pid, "name": name, "pos": pos, "team": team, "opp": opp,
                        "game_id": gm.game_id, "gp": len(cur),
                        "touches": touches, "proj": {}, "naive": {}}
                for stat, (label, poss, kind, miles, gs_exp) in PROP_STATS.items():
                    if pos not in poss:
                        continue
                    base = float(np.average(hist[stat], weights=w))
                    f_opp = def_f.get((opp, stat, pos), 1.0)
                    proj = base * f_opp * ratio ** gs_exp
                    info["proj"][stat] = proj
                    info["naive"][stat] = float(naive_src[stat].mean())
                rows.append(info)
    return rows


def calib(p, k):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return 1 / (1 + math.exp(-k * math.log(p / (1 - p))))


def prop_prob(stat, mean, k):
    """P(stat >= k) for a player projection."""
    kind = PROP_STATS[stat][2]
    if stat == "td":
        return prob_at_least(kind, TD_CALIB[0] + TD_CALIB[1] * mean, 0, k)
    return calib(prob_at_least(kind, mean, prop_sd(stat, mean), k), PROP_CALIB_K)


def prop_dist(stat, mean):
    """(kind, mean, sd) used by the page's line calculator."""
    kind = PROP_STATS[stat][2]
    if stat == "td":
        mean = TD_CALIB[0] + TD_CALIB[1] * mean
    return kind, mean, prop_sd(stat, mean)


# ==================================================================
# WEEK BUILD: projections + every market for one week
# ==================================================================
def early_cushion(week):
    return CUSHION_EARLY if week <= 6 else 0.0


def make_market(gid, market, label, side, p_win, p_push, price, other_price, week, kind="side", line=None):
    """One bettable line. p_win/p_push are model probabilities."""
    p_loss = max(0.0, 1 - p_win - p_push)
    denom = max(1e-9, 1 - p_push)
    pc = p_win / denom                              # win prob given no push
    cush = CUSHION_SIDES + early_cushion(week)
    lo = max(0.001, pc - cush)
    mkt = no_vig(price, other_price) if price is not None and other_price is not None else None
    d = am_to_dec(price)
    ev = (pc * (d - 1) - (1 - pc)) if d else None
    ev_lo = (lo * (d - 1) - (1 - lo)) if d else None
    return {"game_id": gid, "market": market, "label": label, "side": side, "kind": kind,
            "p": pc, "lo": lo, "p_push": p_push, "fair": prob_to_am(pc), "bet_at": prob_to_am(lo),
            "price": price, "mkt": mkt, "ev": ev, "ev_lo": ev_lo, "line": line}


def team_total_probs(mu, line):
    return total_probs(mu, line, TEAM_SCORE_SD)


def build_week(season, week, games, tg, st, md, with_props=True):
    r = fit_ratings(tg, season, week)
    wk_games = games[(games.season == season) & (games.week == week)].sort_values(["gameday", "gametime"])
    team_avg = {t: r["mu"] + r["off"][t] + r["hfa"] / 2 for t in r["teams"]}
    out_games, team_pts = [], {}
    for _, g in wk_games.iterrows():
        neutral = g.location == "Neutral"
        hp, ap = project_game(r, g.home_team, g.away_team, neutral)
        m = (hp - ap) * MARGIN_SCALE
        hp, ap = (hp + ap) / 2 + m / 2, (hp + ap) / 2 - m / 2
        team_pts[g.home_team], team_pts[g.away_team] = hp, ap
        tot = hp + ap
        # prices use the model's view, optionally anchored toward the market line
        mp = m if pd.isna(g.spread_line) else g.spread_line + MODEL_WEIGHT_SIDES * (m - g.spread_line)
        tp = tot if pd.isna(g.total_line) else g.total_line + MODEL_WEIGHT_TOTALS * (tot - g.total_line)
        hpp, app = tp / 2 + mp / 2, tp / 2 - mp / 2
        ks, pm = md.pmf(mp)
        p_home, p_tie = pm[ks > 0].sum(), pm[ks == 0].sum()
        gid = g.game_id
        mk = []
        mk.append(make_market(gid, "Moneyline", f"{g.home_team} ML", g.home_team, p_home, p_tie,
                              g.home_moneyline, g.away_moneyline, week))
        mk.append(make_market(gid, "Moneyline", f"{g.away_team} ML", g.away_team, 1 - p_home - p_tie, p_tie,
                              g.away_moneyline, g.home_moneyline, week))
        if not pd.isna(g.spread_line):
            hl = -g.spread_line
            w_, p_, l_ = md.cover(mp, hl)
            mk.append(make_market(gid, "Spread", f"{g.home_team} {fmt_line(hl)}", g.home_team, w_, p_,
                                  g.home_spread_odds, g.away_spread_odds, week, line=hl))
            mk.append(make_market(gid, "Spread", f"{g.away_team} {fmt_line(-hl)}", g.away_team, l_, p_,
                                  g.away_spread_odds, g.home_spread_odds, week, line=-hl))
        if not pd.isna(g.total_line):
            o, p_, u = total_probs(tp, g.total_line, md.total_sd)
            mk.append(make_market(gid, "Total", f"Over {g.total_line:g}", "over", o, p_,
                                  g.over_odds, g.under_odds, week, line=g.total_line))
            mk.append(make_market(gid, "Total", f"Under {g.total_line:g}", "under", u, p_,
                                  g.under_odds, g.over_odds, week, line=g.total_line))
            # team totals: no price in the feed -> price-check only, at the market-implied line
            imp_h = (g.total_line + g.spread_line) / 2
            for team, mu, imp in ((g.home_team, hpp, imp_h), (g.away_team, app, g.total_line - imp_h)):
                line = math.floor(imp) + 0.5
                o, p_, u = team_total_probs(mu, line)
                mk.append(make_market(gid, "Team total", f"{team} over {line:g}", team + "-o", o, p_, None, None,
                                      week, kind="pc", line=line))
                mk.append(make_market(gid, "Team total", f"{team} under {line:g}", team + "-u", u, p_, None, None,
                                      week, kind="pc", line=line))
        out_games.append({"g": g, "hp": hp, "ap": ap, "m": m, "tot": tot, "p_home": p_home, "p_tie": p_tie,
                          "markets": mk, "props": []})
    props = []
    if with_props and len(wk_games):
        df = defense_factors(st, season, week)
        props = project_players(st, season, week, wk_games, team_pts, team_avg, df)
        by_game = {}
        for p in props:
            by_game.setdefault(p["game_id"], []).append(p)
        for og in out_games:
            og["props"] = by_game.get(og["g"].game_id, [])
    return {"season": season, "week": week, "ratings": r, "games": out_games, "props": props}


def fmt_line(x):
    return "PK" if x == 0 else (f"+{x:g}" if x > 0 else f"{x:g}")


# ==================================================================
# TOP BETS
# ==================================================================
def side_key(mk, g):
    """ML and spread on the same team count as one side; over/under share a total."""
    if mk["market"] in ("Moneyline", "Spread"):
        return "side"
    if mk["market"] == "Total":
        return "total"
    return mk["market"] + mk["side"]


def qualifies(mk):
    if mk["price"] is None or (isinstance(mk["price"], float) and math.isnan(mk["price"])):
        return None
    if mk["p"] < MIN_PROB:
        return None
    if mk["mkt"] is not None and abs(mk["p"] - mk["mkt"]) > MAX_MARKET_GAP:
        return None
    d = am_to_dec(mk["price"])
    if mk["p"] >= HIGH_PROB and mk["ev"] > 0:
        return "hp"
    if mk["ev_lo"] > 0:
        return "va"
    return None


def top_bets(wk):
    hp, va = [], []
    for og in wk["games"]:
        for mk in og["markets"]:
            t = qualifies(mk)
            if t:
                mk["tag"] = t
                (hp if t == "hp" else va).append((og, mk))
    hp.sort(key=lambda x: -x[1]["p"])
    va.sort(key=lambda x: -x[1]["ev_lo"])
    picks, used, per_game = [], set(), {}
    i = j = 0
    turn = "hp"
    while i < len(hp) or j < len(va):
        if (turn == "hp" and i < len(hp)) or j >= len(va):
            og, mk = hp[i]; i += 1
        else:
            og, mk = va[j]; j += 1
        turn = "va" if turn == "hp" else "hp"
        gid = og["g"].game_id
        key = (gid, side_key(mk, og["g"]))
        if key in used or per_game.get(gid, 0) >= MAX_BETS_PER_GAME:
            continue
        used.add(key)
        per_game[gid] = per_game.get(gid, 0) + 1
        picks.append((og, mk))
    return picks


def price_checks(wk, n=12):
    """Markets without a fetched price: team totals + player milestones. Take only at Bet-at or better."""
    items = []
    for og in wk["games"]:
        for mk in og["markets"]:
            if mk["kind"] == "pc" and 0.55 <= mk["p"] <= 0.80:
                items.append({"og": og, "label": mk["label"], "market": "Team total", "p": mk["p"],
                              "lo": max(0.001, mk["p"] - CUSHION_SIDES - early_cushion(wk["week"])),
                              "note": f"Projected {og['g'].away_team} {og['ap']:.1f} - {og['g'].home_team} {og['hp']:.1f}",
                              "stand": abs(mk["p"] - 0.5)})
        for pl in og["props"]:
            for stat, mean in pl["proj"].items():
                nv = pl["naive"].get(stat, 0)
                if nv < MIN_PROP_LINE.get(stat, 0) or pl["gp"] < 3:
                    continue
                boost = mean / nv - 1
                label = PROP_STATS[stat][0]
                if stat == "td":
                    p = prop_prob(stat, mean, 1)
                    if p < 0.40:
                        continue
                    bet = f"{pl['name']} anytime TD"
                else:
                    line = math.floor(nv) + 0.5
                    pr = prop_prob(stat, mean, math.ceil(line))
                    p, bet = (pr, f"{pl['name']} over {line:g} {label.lower()}") if boost > 0 else \
                        (1 - pr, f"{pl['name']} under {line:g} {label.lower()}")
                    if p < 0.55:
                        continue
                cush = CUSHION_PROPS + early_cushion(wk["week"])
                items.append({"og": og, "label": bet, "market": f"Player prop ({pl['team']})", "p": p,
                              "lo": max(0.001, p - cush), "stand": abs(boost) * min(1, pl["gp"] / 4),
                              "note": f"Proj {fmt_stat(stat, mean)} vs {fmt_stat(stat, nv)} season avg"
                                      f" &middot; {pl['gp']} GP &middot; vs {pl['opp']}",
                              "pl": pl, "stat": stat, "line": None if stat == "td" else math.floor(nv) + 0.5,
                              "dir": "over" if (stat == "td" or boost > 0) else "under"})
    items.sort(key=lambda x: -x["stand"])
    seen, out = set(), []
    for it in items:
        k = (it["og"]["g"].game_id, it["label"].split(" over ")[0].split(" under ")[0].split(" anytime")[0])
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
        if len(out) >= n:
            break
    return out


def fmt_stat(stat, v):
    return f"{v:.0f}" if stat.endswith("_yd") else f"{v:.1f}" if stat != "td" else f"{v:.2f} TD/g"


# ==================================================================
# GRADING
# ==================================================================
def grade_market(mk, g):
    """1 = win, 0.5 = push, 0 = loss, None = not played yet."""
    if not g.done:
        return None
    res = g.home_score - g.away_score
    m, side = mk["market"], mk["side"]
    if m == "Moneyline":
        v = res if side == g.home_team else -res
    elif m == "Spread":
        v = (res if side == g.home_team else -res) + mk["line"]
    elif m == "Total":
        v = (g.home_score + g.away_score - mk["line"]) * (1 if side == "over" else -1)
    elif m == "Team total":
        team, ou = side.split("-")
        sc = g.home_score if team == g.home_team else g.away_score
        v = (sc - mk["line"]) * (1 if ou == "o" else -1)
    else:
        return None
    return 1.0 if v > 0 else (0.5 if v == 0 else 0.0)


def profit(outcome, price):
    if outcome is None:
        return None
    if outcome == 0.5:
        return 0.0
    d = am_to_dec(price)
    return STAKE * (d - 1) if outcome == 1 else -STAKE


def grade_week(wk, st, with_props=True):
    """Attach outcomes to markets, top bets, price checks and props."""
    season, week = wk["season"], wk["week"]
    for og in wk["games"]:
        for mk in og["markets"]:
            mk["out"] = grade_market(mk, og["g"])
    wk["picks"] = top_bets(wk)
    for og, mk in wk["picks"]:
        mk["pl"] = profit(mk["out"], mk["price"])
    act = st[(st.season == season) & (st.week == week)].drop_duplicates("player_id").set_index("player_id")
    prows = []
    if with_props:
        for p in wk["props"]:
            if p["player_id"] not in act.index:
                p["actual"] = None
                continue
            a = act.loc[p["player_id"]]
            if a.off_snp <= 0:
                p["actual"] = None     # did not play: void, like a sportsbook
                continue
            p["actual"] = {s: float(a.rush_td + a.rec_td) if s == "td" else float(a[s]) for s in p["proj"]}
            for s, mean in p["proj"].items():
                y = p["actual"][s]
                nv = p["naive"][s]
                row = {"week": week, "stat": s, "proj": mean, "naive": nv, "act": y, "name": p["name"]}
                if s == "td":
                    row.update(p_td=prop_prob(s, mean, 1), y_td=int(y >= 1),
                               p_td_naive=1 - math.exp(-(TD_CALIB[0] + TD_CALIB[1] * nv)))
                else:
                    line = math.floor(nv) + 0.5
                    row.update(p_over=prop_prob(s, mean, math.ceil(line)), y_over=int(y > line))
                    row["miles"] = [(prop_prob(s, mean, k), int(y >= k)) for k in PROP_STATS[s][3]]
                prows.append(row)
    wk["prop_grades"] = prows
    wk["checks"] = price_checks(wk)
    for it in wk["checks"]:
        g = it["og"]["g"]
        it["out"] = None
        if "pl" in it:
            pl = it["pl"]
            if pl.get("actual"):
                y = pl["actual"][it["stat"]]
                it["out"] = float(y >= 1) if it["stat"] == "td" else \
                    float((y > it["line"]) if it["dir"] == "over" else (y < it["line"]))
                it["actual"] = y
        else:
            for mk in it["og"]["markets"]:
                if mk["label"] == it["label"]:
                    it["out"] = mk.get("out")
    return wk


def brier(ps, ys):
    ps, ys = np.asarray(ps, float), np.asarray(ys, float)
    return float(np.mean((ps - ys) ** 2)) if len(ps) else None


def summarize_games(weeks):
    """Season-level game-market metrics from a list of graded weeks."""
    ml_p, ml_m, ml_y, mar_e, spr_e, tot_e, mtot_e = [], [], [], [], [], [], []
    ats, ou, su = [0, 0, 0], [0, 0, 0], [0, 0]
    bets = []
    for wk in weeks:
        for og in wk["games"]:
            g = og["g"]
            if not g.done:
                continue
            res = g.home_score - g.away_score
            if res != 0:
                ml_p.append(og["p_home"] / (1 - og["p_tie"]))
                ml_y.append(1 if res > 0 else 0)
                nv = no_vig(g.home_moneyline, g.away_moneyline)
                ml_m.append(nv if nv is not None else 0.5)
                su[0 if (og["m"] > 0) == (res > 0) else 1] += 1
            mar_e.append(abs(og["m"] - res))
            tot_e.append(abs(og["tot"] - g.home_score - g.away_score))
            if not pd.isna(g.spread_line):
                spr_e.append(abs(g.spread_line - res))
                gap = og["m"] - g.spread_line
                v = res - g.spread_line
                if gap != 0:
                    ats[0 if v * gap > 0 else (2 if v == 0 else 1)] += 1
            if not pd.isna(g.total_line):
                mtot_e.append(abs(g.total_line - g.home_score - g.away_score))
                gap = og["tot"] - g.total_line
                v = g.home_score + g.away_score - g.total_line
                if gap != 0:
                    ou[0 if v * gap > 0 else (2 if v == 0 else 1)] += 1
        bets += [(og, mk) for og, mk in wk.get("picks", []) if mk.get("out") is not None]
    rec = [sum(1 for _, b in bets if b["out"] == 1), sum(1 for _, b in bets if b["out"] == 0),
           sum(1 for _, b in bets if b["out"] == 0.5)]
    units = sum(b["pl"] for _, b in bets)
    return {"n": len(mar_e), "ml_brier": brier(ml_p, ml_y), "mkt_brier": brier(ml_m, ml_y),
            "su": su, "ats": ats, "ou": ou,
            "mar_mae": float(np.mean(mar_e)) if mar_e else None,
            "spr_mae": float(np.mean(spr_e)) if spr_e else None,
            "tot_mae": float(np.mean(tot_e)) if tot_e else None,
            "mtot_mae": float(np.mean(mtot_e)) if mtot_e else None,
            "bets": rec, "units": units, "risked": STAKE * len(bets),
            "hp": [b for _, b in bets if b.get("tag") == "hp"], "va": [b for _, b in bets if b.get("tag") == "va"]}


def summarize_props(weeks):
    rows = [r for wk in weeks for r in wk.get("prop_grades", [])]
    if not rows:
        return None
    d = pd.DataFrame(rows)
    out = {"by_stat": []}
    for s, x in d.groupby("stat"):
        e = {"stat": s, "label": PROP_STATS[s][0], "n": len(x),
             "mae": float((x.act - x.proj).abs().mean()), "naive_mae": float((x.act - x.naive).abs().mean())}
        if s == "td":
            e["brier"] = brier(x.p_td, x.y_td)
            e["naive_brier"] = brier(x.p_td_naive, x.y_td)
            e["hit"] = float(x.y_td.mean())
        else:
            side = (x.p_over > 0.5).astype(int)
            e["ou_hit"] = float((side == x.y_over).mean())
            e["brier"] = brier(x.p_over, x.y_over)
            e["naive_brier"] = brier([0.5] * len(x), x.y_over)
        out["by_stat"].append(e)
    cal = []
    for r in rows:
        if r["stat"] == "td":
            cal.append((r["p_td"], r["y_td"]))
        else:
            cal += r["miles"]
    out["cal"] = cal
    return out


CSS = r"""
:root{--bg:#0e1116;--panel:#161b22;--panel2:#1d2430;--text:#e6edf3;--muted:#8b98a5;--line:#283140;
--accent:#58a6ff;--good:#3fb950;--lean:#d29922;--bad:#f85149;--good-bg:rgba(63,185,80,.15);
--lean-bg:rgba(210,153,34,.15);--bad-bg:rgba(248,81,73,.13);--bar:#3b82f6;--bar2:#f59e0b}
@media (prefers-color-scheme: light){:root{--bg:#f4f6f9;--panel:#fff;--panel2:#f0f3f7;--text:#1b1f24;
--muted:#5d6875;--line:#dfe4ea;--accent:#0969da;--good:#1a7f37;--lean:#9a6700;--bad:#cf222e;
--good-bg:#e3f4e8;--lean-bg:#fff3d1;--bad-bg:#fde7e7;--bar:#2563eb;--bar2:#d97706}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;font-variant-numeric:tabular-nums}
header{max-width:1240px;margin:auto;padding:22px 16px 10px}
h1{margin:0;font-size:22px;letter-spacing:-.01em}
h2{font-size:17px;margin:22px 0 10px}
h3{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin:18px 0 8px}
.sub{color:var(--muted);font-size:13px}
nav{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line)}
nav .in{max-width:1240px;margin:auto;display:flex;gap:2px;padding:0 10px;overflow-x:auto}
nav button{background:none;border:0;color:var(--muted);padding:12px 10px;font:inherit;font-weight:600;border-bottom:2px solid transparent;cursor:pointer;white-space:nowrap}
nav button.on{color:var(--text);border-color:var(--accent)}
main{max-width:1240px;margin:auto;padding:8px 16px 80px}
.tab{display:none}.tab.on{display:block}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;margin:12px 0;overflow:hidden}
details.game>summary{list-style:none;cursor:pointer;padding:14px 16px}
details.game>summary::-webkit-details-marker{display:none}
details.game[open]>summary{border-bottom:1px solid var(--line)}
.gh{display:grid;grid-template-columns:76px 1fr auto;gap:12px;align-items:center}
.time{color:var(--muted);font-size:12px;font-weight:600}
.tline{display:flex;align-items:center;gap:8px;padding:2px 0}
.tline img{width:26px;height:26px}
.tline .ab{font-weight:700;min-width:42px}
.tline .nm{color:var(--muted);font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pill{display:inline-block;white-space:nowrap;font-size:11px;font-weight:700;padding:1px 6px;border-radius:999px;background:var(--panel2);color:var(--muted)}
.pill.warn{background:var(--lean-bg);color:var(--lean)}
.proj{text-align:right}
.proj .sc{font-size:20px;font-weight:700}
.proj .wp{font-size:12px;color:var(--muted)}
.gb{padding:6px 16px 18px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.box{background:var(--panel2);border-radius:10px;padding:12px}
.kv{display:flex;justify-content:space-between;gap:8px;padding:2px 0;font-size:13px}
.kv span:first-child{color:var(--muted)}
.big{font-size:22px;font-weight:700}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:12px;position:sticky;top:0;background:var(--panel)}
td:first-child,th:first-child{text-align:left}
tr.grp td{background:var(--panel2);font-weight:700;font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
td.l{text-align:left}
.good{color:var(--good)}.bad{color:var(--bad)}.lean{color:var(--lean)}
.r-top{color:var(--good);font-weight:600}.r-bot{color:var(--bad)}
.better{font-weight:700}
input.odds{width:72px;padding:4px 6px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--text);font:inherit;text-align:right}
.ev{font-weight:700;font-size:12px;padding:2px 6px;border-radius:6px}
.ev.v{background:var(--good-bg);color:var(--good)}.ev.l{background:var(--lean-bg);color:var(--lean)}.ev.a{background:var(--bad-bg);color:var(--bad)}
.betat{color:var(--good);font-weight:600}
.avoid{color:var(--bad)}
.bars{display:flex;flex-direction:column;gap:4px}
.brow{display:grid;grid-template-columns:52px 1fr 1fr;gap:6px;align-items:center;font-size:12px}
.bcell{display:flex;align-items:center;gap:6px}
.bar{height:10px;border-radius:3px;background:var(--bar)}
.bar.b2{background:var(--bar2)}
.legend{font-size:12px;color:var(--muted);margin:6px 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 4px -1px 8px}
.note{font-size:12px;color:var(--muted);margin-top:8px}
.slate td,.slate th{font-size:13px}
.hot{color:var(--lean);font-weight:700}
.prose{max-width:760px}.prose p,.prose li{color:var(--text)}
th.sort{cursor:pointer}
.pick{display:grid;grid-template-columns:30px 1fr auto;gap:10px;padding:14px 16px;border-bottom:1px solid var(--line);align-items:start}
.pick:last-child{border-bottom:0}
.rk{font-size:18px;font-weight:800;color:var(--muted);line-height:1.2}
.pk-bet{font-size:16px;font-weight:700;margin:3px 0 2px}
.pk-meta{font-size:12px;color:var(--muted)}
.pk-nums{display:flex;flex-wrap:wrap;gap:2px 14px;font-size:12px;margin-top:4px;color:var(--muted)}
.pk-nums b{color:var(--text)}
.pk-right{text-align:right;min-width:96px}
.pk-price{font-size:18px;font-weight:800}
.tag{display:inline-block;font-size:11px;font-weight:700;padding:1px 7px;border-radius:999px;margin-right:6px}
.tag.hp{background:var(--good-bg);color:var(--good)}
.tag.va{background:rgba(88,166,255,.16);color:var(--accent)}
.tag.pc{background:var(--panel2);color:var(--muted)}
.statline{display:flex;flex-wrap:wrap;gap:8px 22px;padding:12px 16px}
.statline div{font-size:12px;color:var(--muted)}.statline b{display:block;font-size:17px;color:var(--text)}
@media (max-width:760px){.pick{grid-template-columns:22px 1fr auto;padding:12px 10px;gap:8px}.pk-right{min-width:84px}
 .grid2{grid-template-columns:1fr}.gh{grid-template-columns:58px 1fr auto}
 .tline .nm{display:none}h1{font-size:19px}th,td{padding:5px 6px}.gb{padding:6px 10px 16px}main{padding:8px 10px 80px}}

.res{display:inline-block;font-size:11px;font-weight:800;padding:1px 7px;border-radius:999px}
.res.w{background:var(--good-bg);color:var(--good)}.res.l{background:var(--bad-bg);color:var(--bad)}.res.p{background:var(--panel2);color:var(--muted)}
.final{font-size:12px;font-weight:700;color:var(--text)}
.backlink{color:var(--accent);text-decoration:none;font-size:13px;font-weight:600}
select.calc,input.calc{padding:4px 6px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--text);font:inherit}
input.calc{width:72px;text-align:right}
.calcrow{display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:13px}
.calcout{font-size:13px;margin-top:6px}
.warnbox{background:var(--lean-bg);color:var(--lean);border-radius:10px;padding:10px 12px;font-size:13px;margin:10px 0}
"""


# ==================================================================
# HTML
# ==================================================================
E_ = html.escape
ESPN = {"WAS": "wsh", "LA": "lar"}


def logo(t):
    code = ESPN.get(t, t.lower())
    return (f'<img src="https://a.espncdn.com/i/teamlogos/nfl/500/{code}.png" alt="" loading="lazy" '
            f'onerror="this.style.display=\'none\'">')


def pct(p, d=0):
    return "-" if p is None else f"{100 * p:.{d}f}%"


def kickoff(g):
    try:
        dt = datetime.strptime(f"{g.gameday} {g.gametime}", "%Y-%m-%d %H:%M")
        return dt.strftime("%a %-I:%M %p")
    except Exception:
        return str(g.gameday)


def res_badge(out):
    if out is None:
        return ""
    return {1.0: '<span class="res w">WIN</span>', 0.0: '<span class="res l">LOSS</span>',
            0.5: '<span class="res p">PUSH</span>'}[out]


def records_before(games, season, week):
    d = games[(games.season == season) & (games.week < week) & games.done & (games.game_type == "REG")]
    rec = {}
    for _, g in d.iterrows():
        for t, s, o in ((g.home_team, g.home_score, g.away_score), (g.away_team, g.away_score, g.home_score)):
            r = rec.setdefault(t, [0, 0, 0])
            r[0 if s > o else (1 if s < o else 2)] += 1
    return rec


def fmt_rec(r):
    if not r:
        return "0-0"
    return f"{r[0]}-{r[1]}" + (f"-{r[2]}" if r[2] else "")


def odds_cell(mk, graded):
    if graded and mk.get("out") is not None:
        price = fmt_am(mk["price"]) if mk["price"] is not None and not pd.isna(mk["price"]) else "-"
        return f'<td>{price}<div>{res_badge(mk["out"])}</div></td>'
    val = "" if mk["price"] is None or pd.isna(mk["price"]) else fmt_am(mk["price"])
    return (f'<td><input class="odds" inputmode="text" placeholder="odds" aria-label="Your book odds" '
            f'value="{val}" title="Closing price (nflverse)"><div><span class="ev"></span></div></td>')


def render_pick(i, og, mk):
    g = og["g"]
    tag = '<span class="tag hp">High probability</span>' if mk["tag"] == "hp" else '<span class="tag va">Value</span>'
    right = (f'<div class="pk-price">{fmt_am(mk["price"])}</div>'
             f'<div class="pk-meta">{"Closing line" if g.done else "Latest line"}</div>')
    if mk.get("out") is not None:
        pl = mk["pl"]
        right += f'<div style="margin-top:6px">{res_badge(mk["out"])}</div>' \
                 f'<div class="pk-meta">{pl:+.2f}u</div>'
    else:
        right += ('<div style="margin-top:6px"><input class="odds" placeholder="price" aria-label="Your price">'
                  '<div><span class="ev"></span></div></div>')
    final = f' &middot; Final {g.away_team} {g.away_score:.0f} - {g.home_team} {g.home_score:.0f}' if g.done else ""
    return (f'<div class="pick" data-p="{mk["p"]:.4f}" data-lo="{mk["lo"]:.4f}"><div class="rk">{i}</div><div>'
            f'<div>{tag}<span class="pk-meta">{kickoff(g)} &middot; {g.away_team} @ {g.home_team} &middot; {mk["market"]}</span></div>'
            f'<div class="pk-bet">{E_(mk["label"])}</div><div class="pk-nums"><span>Model <b>{pct(mk["p"])}</b></span>'
            f'<span>Market <b>{pct(mk["mkt"])}</b></span><span>Fair <b>{fmt_am(mk["fair"])}</b></span>'
            f'<span>Bet at <b class="good">{fmt_am(mk["bet_at"])}</b> or better</span></div>'
            f'<div class="note" style="margin-top:4px">Projected {g.away_team} {og["ap"]:.1f} - {g.home_team} {og["hp"]:.1f}'
            f' &middot; market spread {fmt_line(-g.spread_line)} {g.home_team}, total {g.total_line:g}{final}</div></div>'
            f'<div class="pk-right">{right}</div></div>')


def render_check(i, it):
    g = it["og"]["g"]
    right = f'<div class="pk-meta">Bet at</div><div class="pk-price good">{fmt_am(prob_to_am(it["lo"]))}</div>'
    if it.get("out") is not None:
        badge = {1.0: '<span class="res w">HIT</span>', 0.5: '<span class="res p">PUSH</span>'}.get(
            it["out"], '<span class="res l">MISS</span>')
        act = f'actual {it["actual"]:g}' if it.get("actual") is not None else ""
        right += f'<div style="margin-top:6px">{badge}</div><div class="pk-meta">{act}</div>'
    else:
        right += ('<div style="margin-top:6px"><input class="odds" placeholder="price" aria-label="Your price">'
                  '<div><span class="ev"></span></div></div>')
    return (f'<div class="pick" data-p="{it["p"]:.4f}" data-lo="{it["lo"]:.4f}"><div class="rk">{i}</div><div>'
            f'<div><span class="tag pc">Price check</span><span class="pk-meta">{kickoff(g)} &middot; {g.away_team} @ {g.home_team} &middot; {it["market"]}</span></div>'
            f'<div class="pk-bet">{E_(it["label"])}</div><div class="pk-nums"><span>Model <b>{pct(it["p"])}</b></span>'
            f'<span>Fair <b>{fmt_am(prob_to_am(it["p"]))}</b></span><span>Bet at <b class="good">{fmt_am(prob_to_am(it["lo"]))}</b> or better</span></div>'
            f'<div class="note" style="margin-top:4px">{it["note"]}</div></div><div class="pk-right">{right}</div></div>')


def long_run_note(long_run):
    if not long_run:
        return ""
    sm = summarize_games([w for ws in long_run.values() for w in ws])
    roi = sm["units"] / sm["risked"] if sm["risked"] else 0
    yrs = f"{min(long_run)}-{max(long_run)}"
    return (f'<div class="warnbox"><b>Read this first.</b> Re-run under the same rules for {yrs}, the top bets went '
            f'<b>{wlp(sm["bets"])}</b> ({sm["units"]:+.1f}u, {roi:+.1%} ROI) and the model\'s side of every closing spread '
            f'went {wlp(sm["ats"])} ({hit_pct(sm["ats"])}). That is break-even: against <i>closing</i> lines the model has '
            f'not shown an edge on game markets yet. Treat picks as leans, bet them only at the <b>Bet at</b> price or better, '
            f'and get the most out of the board by shopping early lines and prices. Player projections did beat season '
            f'averages in every stat (Track record tab).</div>')


def render_top(wk_list, note=""):
    out = [note, '<h2>Top bets</h2><p class="sub" style="max-width:820px">Picks come from the moneyline, spread and total, '
           'alternating two kinds: <span class="tag hp">High probability</span> bets the model gives '
           f'{HIGH_PROB:.0%}+ where the price beats fair odds, and <span class="tag va">Value</span> bets where the price '
           'beats the cautious <b>Bet at</b> price. At most 2 bets per game, never two on the same side or total, and '
           f'never when the model is more than {MAX_MARKET_GAP:.0%} away from the market. Prices are the closing lines '
           'from nflverse; type your own book\'s price to re-check.</p>']
    for wk, title in wk_list:
        picks = wk["picks"]
        out.append(f'<h2>{title}</h2>')
        if not picks:
            out.append('<p class="sub">No game bet clears the bar this week.</p>')
        else:
            graded = [mk for _, mk in picks if mk.get("out") is not None]
            if graded:
                w = sum(1 for m in graded if m["out"] == 1); l = sum(1 for m in graded if m["out"] == 0)
                p = sum(1 for m in graded if m["out"] == 0.5); u = sum(m["pl"] for m in graded)
                out.append(f'<p class="sub">Graded: <b>{w}-{l}{"-" + str(p) if p else ""}</b> &middot; '
                           f'<b class="{"good" if u >= 0 else "bad"}">{u:+.2f} units</b> at closing prices'
                           f'{" &middot; " + str(len(picks) - len(graded)) + " still to play" if len(graded) < len(picks) else ""}</p>')
            out.append('<div class="card">' + "".join(render_pick(i + 1, og, mk) for i, (og, mk) in enumerate(picks)) + '</div>')
        if wk.get("checks"):
            out.append('<h3>Price check</h3><p class="sub" style="max-width:820px">No sportsbook price in the feed '
                       '(team totals, player props). Lines shown are the market-implied team total or the player\'s '
                       'season average, rounded. Ranked by how far this week\'s matchup moves the projection. '
                       'Take them only at the <b>Bet at</b> price or better, and check the book\'s actual line first.</p>')
            out.append('<div class="card">' + "".join(render_check(i + 1, it) for i, it in enumerate(wk["checks"])) + '</div>')
    return "".join(out)


def render_slate(wk):
    rows = []
    for og in wk["games"]:
        g = og["g"]
        ph = og["p_home"] / (1 - og["p_tie"])
        fin = f'{g.away_score:.0f} - {g.home_score:.0f}' if g.done else "-"
        rows.append(f'<tr><td class="l">{kickoff(g)}</td><td class="l"><b>{g.away_team}</b> @ <b>{g.home_team}</b></td>'
                    f'<td>{og["ap"]:.1f} - {og["hp"]:.1f}</td><td>{pct(1 - ph)} / {pct(ph)}</td>'
                    f'<td>{fmt_line(-round(og["m"] * 2) / 2)} / {fmt_line(-g.spread_line)}</td>'
                    f'<td>{og["tot"]:.1f} / {g.total_line:g}</td><td>{fin}</td></tr>')
    return ('<div class="card"><div class="tw"><table class="slate"><thead><tr><th>Kickoff</th><th>Matchup</th>'
            '<th>Proj.</th><th>Win %</th><th>Home spread (model / mkt)</th><th>Total (model / mkt)</th><th>Final</th>'
            '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div></div>')


def render_board(og, graded):
    groups = {}
    for mk in og["markets"]:
        groups.setdefault(mk["market"], []).append(mk)
    names = {"Moneyline": "Moneyline (tie = push)", "Spread": "Spread", "Total": "Game total",
             "Team total": "Team totals (no feed price - type yours)"}
    rows = []
    for m in ("Moneyline", "Spread", "Total", "Team total"):
        if m not in groups:
            continue
        rows.append(f'<tr class="grp"><td colspan="5">{names[m]}</td></tr>')
        for mk in groups[m]:
            push = f'<div style="font-size:11px;color:var(--muted)">push {pct(mk["p_push"])}</div>' if mk["p_push"] > 0.02 else ""
            rows.append(f'<tr data-p="{mk["p"]:.4f}" data-lo="{mk["lo"]:.4f}"><td class="l">{E_(mk["label"])}</td>'
                        f'<td>{pct(mk["p"], 1)}{push}</td><td><b>{fmt_am(mk["fair"])}</b><div class="avoid" style="font-size:11px">avoid worse</div></td>'
                        f'<td class="betat">{fmt_am(mk["bet_at"])}<div style="font-size:11px;color:var(--muted)">or better</div></td>'
                        f'{odds_cell(mk, graded)}</tr>')
    return ('<h3>Betting board</h3><div class="tw"><table><thead><tr><th>Market</th><th>Model</th><th>Fair odds</th>'
            f'<th>Bet at</th><th>{"Close / result" if graded else "Your book"}</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>')


def render_props(og, graded, week):
    props = sorted(og["props"], key=lambda p: (p["team"] != og["g"].away_team, ["QB", "RB", "WR", "TE"].index(p["pos"]), -p["touches"]))
    if not props:
        return ""
    cols = [("pass_yd", "Pass yds"), ("pass_td", "Pass TD"), ("rush_yd", "Rush yds"), ("rec", "Rec"),
            ("rec_yd", "Rec yds"), ("td", "Anytime TD")]
    head = "".join(f"<th>{c[1]}</th>" for c in cols)
    rows, opts = [], []
    team = None
    for p in props:
        if p["team"] != team:
            team = p["team"]
            rows.append(f'<tr class="grp"><td colspan="{len(cols) + 1}">{team} vs {p["opp"]}</td></tr>')
        cells = []
        for s, lab in cols:
            if s not in p["proj"]:
                cells.append("<td></td>")
                continue
            v = p["proj"][s]
            txt = pct(prop_prob(s, v, 1)) if s == "td" else (f"{v:.0f}" if s.endswith("yd") else f"{v:.1f}")
            act = ""
            if graded and p.get("actual"):
                a = p["actual"][s]
                act = f'<div style="font-size:11px;color:var(--muted)">{"scored" if (s == "td" and a >= 1) else ("no TD" if s == "td" else f"{a:g}")}</div>'
            cells.append(f"<td>{txt}{act}</td>")
            kind, mean, sd = prop_dist(s, v)
            if s == "td":
                continue
            opts.append(f'<option data-k="{kind}" data-m="{mean:.3f}" data-sd="{sd:.3f}">{E_(p["name"])} &middot; {lab}'
                        f' (proj {txt})</option>')
        dnp = ' <span class="pill">DNP</span>' if graded and og["g"].done and not p.get("actual") else ""
        rows.append(f'<tr><td class="l">{E_(p["name"])} <span class="sub">{p["pos"]}</span>{dnp}</td>{"".join(cells)}</tr>')
    calc = ""
    if opts:
        calc = (f'<div class="box" style="margin-top:12px"><div class="sub">Prop line check: pick a prop, type the book\'s line and prices</div>'
                f'<div class="calc" data-c="{CUSHION_PROPS + early_cushion(week):.3f}" data-cal="{PROP_CALIB_K}" style="margin-top:6px"><div class="calcrow">'
                f'<select class="calc">{"".join(opts)}</select><span>line</span><input class="calc ln" placeholder="e.g. 54.5">'
                f'<span>over</span><input class="calc po" placeholder="-115"><span>under</span><input class="calc pu" placeholder="-115">'
                f'</div><div class="calcout"></div></div></div>')
    note = '<div class="note">Projections, not sportsbook lines. Grey numbers under a projection are what happened.</div>' if graded else \
        '<div class="note">Players who saw the field in either of the team\'s last two games. Check injury reports before betting.</div>'
    return (f'<h3>Player projections</h3><div class="tw"><table><thead><tr><th>Player</th>{head}</tr></thead><tbody>'
            + "".join(rows) + f'</tbody></table></div>{note}{calc}')


def render_game(og, rec, r, graded, week):
    g = og["g"]
    ph = og["p_home"] / (1 - og["p_tie"])
    pa = 1 - ph
    teams = []
    for t, p in ((g.away_team, pa), (g.home_team, ph)):
        sc = ""
        if g.done:
            sc = f'<span class="final" style="margin-left:8px">{(g.away_score if t == g.away_team else g.home_score):.0f}</span>'
        teams.append(f'<div class="tline">{logo(t)}<span class="ab">{t}</span><span class="pill">{fmt_rec(rec.get(t))}</span>'
                     f'<span class="nm">{TEAM_NAMES.get(t, "")}</span>{sc}<span style="margin-left:auto;font-weight:700">{pct(p)}</span></div>')
    qb = ""
    if isinstance(g.get("away_qb_name"), str):
        qb = f'<div class="kv"><span>Starting QBs (feed)</span><b>{E_(g.away_qb_name)} &middot; {E_(str(g.home_qb_name))}</b></div>'
    box1 = (f'<div class="box"><div class="sub">Projected score ({g.away_team} @ {g.home_team})</div>'
            f'<div class="big">{og["ap"]:.1f} - {og["hp"]:.1f}</div>'
            f'<div class="kv"><span>Win probability</span><b>{g.away_team} {pct(pa, 1)} &middot; {g.home_team} {pct(ph, 1)}</b></div>'
            f'<div class="kv"><span>Fair moneyline</span><b>{g.away_team} {fmt_am(prob_to_am(pa))} &middot; {g.home_team} {fmt_am(prob_to_am(ph))}</b></div>'
            f'<div class="kv"><span>Model spread vs market</span><b>{g.home_team} {fmt_line(-round(og["m"] * 2) / 2)} vs {fmt_line(-g.spread_line)}</b></div>'
            f'<div class="kv"><span>Model total vs market</span><b>{og["tot"]:.1f} vs {g.total_line:g}</b></div>'
            f'<div class="kv"><span>Off / Def rating (pts vs avg)</span><b>{g.away_team} {r["off"].get(g.away_team, 0):+.1f}/{r["def"].get(g.away_team, 0):+.1f} &middot; '
            f'{g.home_team} {r["off"].get(g.home_team, 0):+.1f}/{r["def"].get(g.home_team, 0):+.1f}</b></div>{qb}</div>')
    if g.done:
        res = g.home_score - g.away_score
        ats_side = g.home_team if res - g.spread_line > 0 else (g.away_team if res - g.spread_line < 0 else "Push")
        tot = g.home_score + g.away_score
        ou = "Over" if tot > g.total_line else ("Under" if tot < g.total_line else "Push")
        model_side = g.home_team if og["m"] > g.spread_line else g.away_team
        model_tot = "Over" if og["tot"] > g.total_line else "Under"
        box2 = (f'<div class="box"><div class="sub">Result</div><div class="big">{g.away_team} {g.away_score:.0f} - {g.home_team} {g.home_score:.0f}</div>'
                f'<div class="kv"><span>Covered</span><b>{ats_side} (model leaned {model_side}) {res_badge(1.0 if ats_side == model_side else (0.5 if ats_side == "Push" else 0.0))}</b></div>'
                f'<div class="kv"><span>Total {tot:.0f} vs {g.total_line:g}</span><b>{ou} (model leaned {model_tot}) {res_badge(1.0 if ou == model_tot else (0.5 if ou == "Push" else 0.0))}</b></div>'
                f'<div class="kv"><span>Projection miss</span><b>margin {abs(og["m"] - res):.1f} &middot; market {abs(g.spread_line - res):.1f}</b></div></div>')
    else:
        box2 = ('<div class="box"><div class="sub">Reading the odds</div><div class="note" style="margin-top:4px">'
                '<b class="good">Bet at</b> is the price where the bet is still +EV even if the model is a bit too optimistic. '
                'Between <b>Bet at</b> and <b>Fair</b> is a lean. Anything worse than <b>Fair</b>, skip. The box is pre-filled '
                'with the closing price from nflverse; type your sportsbook\'s price to see the edge.</div></div>')
    when = kickoff(g)
    return (f'<details class="card game"><summary><div class="gh"><div class="time">{when.replace(" ", "<br>", 1)}</div>'
            f'<div>{"".join(teams)}</div><div class="proj"><div class="sc">{og["ap"]:.1f} - {og["hp"]:.1f}</div>'
            f'<div class="wp">total {og["tot"]:.1f}</div></div></div></summary><div class="gb"><div class="grid2">{box1}{box2}</div>'
            f'{render_board(og, graded)}{render_props(og, graded, week)}</div></details>')


def render_week(wk, games, title, sub):
    rec = records_before(games, wk["season"], wk["week"])
    graded = any(og["g"].done for og in wk["games"])
    return (f'<h2>{title}</h2><p class="sub">{sub}</p>{render_slate(wk)}'
            + "".join(render_game(og, rec, wk["ratings"], graded, wk["week"]) for og in wk["games"]))


def render_league(games, tg, r, season, week):
    d = games[(games.season == season) & (games.week < week) & games.done & (games.game_type == "REG")]
    t = tg[(tg.season == season) & (tg.week < week)]
    rows = []
    rank = sorted(r["teams"], key=lambda x: -(r["off"][x] - r["def"][x]))
    for team in r["teams"]:
        if team not in TEAM_NAMES:
            continue
        w = l = ti = aw = al = ap = ow = ou = op = 0
        for _, g in d.iterrows():
            if team not in (g.home_team, g.away_team):
                continue
            home = team == g.home_team
            mar = (g.home_score - g.away_score) * (1 if home else -1)
            w += mar > 0; l += mar < 0; ti += mar == 0
            if not pd.isna(g.spread_line):
                v = mar + (-g.spread_line if home else g.spread_line)
                aw += v > 0; al += v < 0; ap += v == 0
            if not pd.isna(g.total_line):
                v = g.home_score + g.away_score - g.total_line
                ow += v > 0; ou += v < 0; op += v == 0
        tt = t[t.team == team]
        gp = max(len(tt), 1)
        ypg = tt.off_yd.mean() if tt.has_box.any() else float("nan")
        opp = t[t.opp == team]
        yapg = opp.off_yd.mean() if opp.has_box.any() else float("nan")
        o, df_ = r["off"][team], r["def"][team]
        net = o - df_
        cls = lambda v, hi: "r-top" if (v > 2.5 if hi else v < -2.5) else ("r-bot" if (v < -2.5 if hi else v > 2.5) else "")
        rows.append(
            f'<tr><td class="l" data-v="{team}"><b>{team}</b> <span class="sub">{TEAM_NAMES[team]}</span></td>'
            f'<td class="l">{TEAM_DIV.get(team, "")}</td><td data-v="{rank.index(team) + 1}">{rank.index(team) + 1}</td>'
            f'<td class="l" data-v="{w + 0.5 * ti}">{w}-{l}{"-" + str(ti) if ti else ""}</td>'
            f'<td class="l" data-v="{aw - al}">{aw}-{al}{"-" + str(ap) if ap else ""}</td>'
            f'<td class="l" data-v="{ow - ou}">{ow}-{ou}{"-" + str(op) if op else ""}</td>'
            f'<td data-v="{tt.pf.mean() if len(tt) else 0:.3f}">{tt.pf.mean() if len(tt) else 0:.1f}</td>'
            f'<td data-v="{tt.pa.mean() if len(tt) else 0:.3f}">{tt.pa.mean() if len(tt) else 0:.1f}</td>'
            f'<td data-v="{0 if math.isnan(ypg) else ypg:.1f}">{"-" if math.isnan(ypg) else f"{ypg:.0f}"}</td>'
            f'<td data-v="{0 if math.isnan(yapg) else yapg:.1f}">{"-" if math.isnan(yapg) else f"{yapg:.0f}"}</td>'
            f'<td data-v="{o:.3f}" class="{cls(o, True)}">{o:+.1f}</td>'
            f'<td data-v="{df_:.3f}" class="{cls(df_, False)}">{df_:+.1f}</td>'
            f'<td data-v="{net:.3f}"><b>{net:+.1f}</b></td></tr>')
    rows.sort(key=lambda x: int(x.split('data-v="')[2].split('"')[0]))
    return ('<h2>League</h2><p class="sub">Tap a column to sort. Off / Def are the model\'s current ratings in points per game '
            'versus an average opponent (Off above 0 is good, Def below 0 is good). Net = Off - Def, roughly the spread '
            f'against an average team on a neutral field. Home field is worth {r["hfa"]:.1f} pts in this fit. '
            f'Records, ATS and O/U are {season} regular season through week {week - 1}.</p>'
            '<div class="card"><div class="tw"><table class="sortable"><thead><tr><th>Team</th><th>Division</th><th>Power</th>'
            '<th>Record</th><th>ATS</th><th>O/U</th><th>PF/G</th><th>PA/G</th><th>Yds/G</th><th>Yds allowed/G</th>'
            '<th>Off</th><th>Def</th><th>Net</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div></div>')


def units_txt(mk):
    return "" if mk.get("pl") is None else f'{mk["pl"]:+.2f}u'


def wlp(x):
    return f"{x[0]}-{x[1]}" + (f"-{x[2]}" if len(x) > 2 and x[2] else "")


def hit_pct(x):
    n = x[0] + x[1]
    return f"{100 * x[0] / n:.0f}%" if n else "-"


def skill(b, mb):
    if b is None or mb is None:
        return "-"
    v = (mb - b) / mb
    return f'<span class="{"good" if v > 0 else "bad"}">{v:+.1%}</span>'


def summary_row(label, s):
    roi = s["units"] / s["risked"] if s["risked"] else 0
    return (f'<tr><td class="l">{label}</td><td>{s["n"]}</td><td>{wlp(s["bets"])}</td>'
            f'<td class="{"good" if s["units"] >= 0 else "bad"}">{s["units"]:+.1f}u</td><td>{roi:+.1%}</td>'
            f'<td>{wlp(s["ats"])} <span class="sub">{hit_pct(s["ats"])}</span></td>'
            f'<td>{wlp(s["ou"])} <span class="sub">{hit_pct(s["ou"])}</span></td>'
            f'<td>{wlp(s["su"])} <span class="sub">{hit_pct(s["su"])}</span></td>'
            f'<td>{s["ml_brier"]:.3f} / {s["mkt_brier"]:.3f}</td>'
            f'<td>{s["mar_mae"]:.1f} / {s["spr_mae"]:.1f}</td><td>{s["tot_mae"]:.1f} / {s["mtot_mae"]:.1f}</td></tr>')


SUMMARY_HEAD = ('<thead><tr><th>Span</th><th>Games</th><th>Top bets</th><th>Units</th><th>ROI</th><th>Every game ATS</th>'
                '<th>Every game O/U</th><th>Straight up</th><th>ML Brier model / mkt</th><th>Margin miss model / mkt</th>'
                '<th>Total miss model / mkt</th></tr></thead>')


def calib_table(pairs):
    bins = [(0, .2), (.2, .35), (.35, .5), (.5, .65), (.65, .8), (.8, 1.01)]
    rows = []
    for lo, hi in bins:
        x = [(p, y) for p, y in pairs if lo <= p < hi]
        if not x:
            continue
        rows.append(f'<tr><td class="l">{lo:.0%}-{min(hi, 1):.0%}</td><td>{len(x)}</td>'
                    f'<td>{np.mean([p for p, _ in x]):.1%}</td><td>{np.mean([y for _, y in x]):.1%}</td></tr>')
    return ('<div class="card"><div class="tw"><table><thead><tr><th>Model said</th><th>Count</th><th>Avg predicted</th>'
            '<th>Actually happened</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div></div>')


def render_track(target, weeks, long_run):
    s_t = summarize_games([target])
    s_all = summarize_games(weeks)
    roi = s_t["units"] / s_t["risked"] if s_t["risked"] else 0
    out = [long_run_note(long_run), f'<h2>Track record</h2><p class="sub" style="max-width:860px">Every week below was predicted <b>walk-forward</b>: '
           f'the model was rebuilt using only games and stats from before that week, then graded against the final '
           f'scores and the closing lines. Nothing from a week leaks into its own predictions. '
           f'Week {target["week"]} is the headline; earlier weeks of {target["season"]} and the 2023-2025 seasons '
           f'show whether it holds up over a bigger sample.</p>']
    pend = sum(1 for og in target["games"] if not og["g"].done)
    out.append(f'<h2>Week {target["week"]} at a glance</h2><div class="card"><div class="statline">'
               f'<div>Top bets<b>{wlp(s_t["bets"])}</b></div>'
               f'<div>Units (1u each)<b class="{"good" if s_t["units"] >= 0 else "bad"}">{s_t["units"]:+.2f}</b></div>'
               f'<div>ROI<b>{roi:+.1%}</b></div>'
               f'<div>Every game ATS<b>{wlp(s_t["ats"])}</b></div>'
               f'<div>Every game O/U<b>{wlp(s_t["ou"])}</b></div>'
               f'<div>Straight up<b>{wlp(s_t["su"])}</b></div>'
               f'<div>ML Brier (model / market)<b>{s_t["ml_brier"]:.3f} / {s_t["mkt_brier"]:.3f}</b></div>'
               f'<div>Games graded<b>{s_t["n"]}{f" &middot; {pend} pending" if pend else ""}</b></div></div></div>')
    bet_rows = []
    for og, mk in target["picks"]:
        g = og["g"]
        bet_rows.append(f'<tr><td class="l">{g.away_team} @ {g.home_team}</td><td class="l">{E_(mk["label"])}</td>'
                        f'<td class="l">{"High prob" if mk["tag"] == "hp" else "Value"}</td><td>{pct(mk["p"])}</td>'
                        f'<td>{pct(mk["mkt"])}</td><td>{fmt_am(mk["price"])}</td>'
                        f'<td>{res_badge(mk.get("out")) or "pending"}</td>'
                        f'<td>{units_txt(mk)}</td></tr>')
    if bet_rows:
        out.append('<div class="card"><div class="tw"><table><thead><tr><th>Game</th><th>Bet</th><th>Type</th><th>Model</th>'
                   '<th>Market</th><th>Close</th><th>Result</th><th>Units</th></tr></thead><tbody>' + "".join(bet_rows) +
                   '</tbody></table></div></div>')
    rows = [summary_row(f'Week {w["week"]}', summarize_games([w])) for w in weeks]
    rows.append(summary_row(f'<b>{target["season"]} weeks {weeks[0]["week"]}-{weeks[-1]["week"]}</b>', s_all))
    out.append(f'<h2>By week ({target["season"]})</h2><p class="sub" style="max-width:860px">"Every game" columns take the '
               'model\'s side of every closing spread and total, no filter. Brier: lower is better; the market column is '
               'the no-vig closing moneyline. Miss = average points between the projection and what happened.</p>'
               f'<div class="card"><div class="tw"><table>{SUMMARY_HEAD}<tbody>{"".join(rows)}</tbody></table></div></div>')
    if long_run:
        rows = [summary_row(f"{season} weeks 2-18", summarize_games(ws)) for season, ws in long_run.items()]
        allw = [w for ws in long_run.values() for w in ws]
        rows.append(summary_row("<b>2023-2025 combined</b>", summarize_games(allw)))
        out.append('<h2>Long-run check</h2><p class="sub" style="max-width:860px">Same model, same rules, rebuilt week by week '
                   'for three full seasons. This is the honest baseline: against closing lines the model sits close to '
                   'break-even, which is normal - closing NFL lines are the sharpest number there is. The edge, if any, '
                   'comes from betting earlier lines that haven\'t moved yet and from shopping prices.</p>'
                   f'<div class="card"><div class="tw"><table>{SUMMARY_HEAD}<tbody>{"".join(rows)}</tbody></table></div></div>')
    ps = summarize_props(weeks)
    pt = summarize_props([target])
    if ps:
        def prow(e, label):
            extra = f'{e["brier"]:.3f} / {e["naive_brier"]:.3f}' if e["stat"] == "td" else "-"
            better = e["mae"] < e["naive_mae"]
            return (f'<tr><td class="l">{label}</td><td>{e["n"]}</td><td class="{"good" if better else "bad"}">{e["mae"]:.2f}</td>'
                    f'<td>{e["naive_mae"]:.2f}</td><td>{extra}</td></tr>')
        rows = []
        for e in (pt or {"by_stat": []})["by_stat"]:
            rows.append(prow(e, f'Week {target["week"]} &middot; {e["label"]}'))
        for e in ps["by_stat"]:
            rows.append(prow(e, f'Weeks {weeks[0]["week"]}-{weeks[-1]["week"]} &middot; {e["label"]}'))
        out.append('<h2>Player projections</h2><p class="sub" style="max-width:860px">Graded only for players who played '
                   '(a DNP voids, like a sportsbook). Compared with the simplest alternative, the player\'s season average. '
                   'Green = the model\'s projection missed by less. Anytime TD is scored with Brier (lower is better). '
                   'There are no historical prop prices in the data, so props are graded on accuracy, not profit.</p>'
                   '<div class="card"><div class="tw"><table><thead><tr><th>Stat</th><th>Graded</th><th>Avg miss (model)</th>'
                   '<th>Avg miss (season avg)</th><th>TD Brier model / avg</th></tr></thead><tbody>' + "".join(rows) +
                   '</tbody></table></div></div>')
    pairs = []
    for wk in weeks + [w for ws in (long_run or {}).values() for w in ws]:
        for og in wk["games"]:
            for mk in og["markets"]:
                if mk.get("out") in (0.0, 1.0) and mk["market"] in ("Moneyline", "Spread", "Total"):
                    pairs.append((mk["p"], mk["out"]))
    out.append('<h2>Calibration</h2><p class="sub">When the model says X%, how often did it happen? Game markets '
               '(both sides of every moneyline, spread and total, pushes removed), all seasons above.</p>' + calib_table(pairs))
    if ps:
        out.append(f'<h3>Player milestones and anytime TD ({target["season"]})</h3>' + calib_table(ps["cal"]))
    return "".join(out)


HOW = """<h2>How the model works</h2>
<p><b>Data.</b> Scores, kickoff times and closing sportsbook lines (spread, total, moneyline and their prices) come from
nflverse's public games file. Player and team box scores come from the Sleeper stats files in this repo's
<code>data/</code> folder (2023 onward), including Sleeper's TEAM rows (total yards, first downs, turnovers).</p>
<p><b>Expected points.</b> Raw points are noisy - a pick-six or a missed field goal swings them. A regression fitted on
past seasons turns each game's yards, first downs and turnovers into expected points. Ratings use
POINTS_BLEND% actual points and the rest expected points.</p>
<p><b>Ratings.</b> Every team gets an offense and a defense number from a weighted regression
(points = league average + offense + opponent's defense + home field), so scoring 30 on a great defense counts for more
than 30 on a bad one. A game's weight halves every HALF_LIFE weeks, last season counts LAST_SEASON as much
and fades as this season builds, and every rating is mixed with RIDGE games of league-average play so one blowout
doesn't take over. Home field is fitted from the data each week; neutral-site games get none.</p>
<p><b>Margin and totals.</b> NFL scores bunch on 3 and 7, so the margin isn't a smooth bell curve. For a projected
margin, the model looks up how real games with a similar closing spread finished (2010 on), which keeps the key numbers -
that's where the push percentages come from. Totals use a bell curve with the historical spread of totals (about
TOTAL_SD points). Team totals use a 9-point spread around each team's projection.</p>
<p><b>Player props.</b> Each player's per-game rate is a recency-weighted average of the games he played (half-life
PHL games; last season counts half). It's adjusted by how much this week's opponent allows to his position (shrunk toward
average) and by game script: a team projected to score more than usual throws, runs and scores a bit more. Yardage uses
a bell curve whose width was fitted on 2024-2025 misses; receptions and touchdowns use Poisson counts. Anytime TD
chances are pulled toward average because raw TD rates over a few games are mostly luck. QBs: the feed's listed starter
is used when present.</p>
<p><b>Fair odds and Bet at.</b> Fair odds are the no-vig price of the model's probability (with pushes removed).
<b>Bet at</b> knocks the probability down by a cushion (CUSH_S points for sides and totals, CUSH_P for props, plus
CUSH_E more through week 6), so a price at or better than it is still a good bet if the model is somewhat off.</p>
<p><b>Top bets.</b> A high-probability bet (HIGH_PROB+) qualifies if the price beats fair odds; any other bet needs the
cautious probability to beat the price. No longshots (model under MIN_PROB) and nothing more than GAP points away from
the market's no-vig probability - in the NFL a big disagreement with the closing line usually means the model is missing
news (injuries, a QB change, weather). Qualifying bets alternate between high-probability picks and the biggest edges.</p>
<p><b>Track record.</b> Each run rebuilds every earlier week of the season walk-forward and grades it, plus three full
past seasons under the same rules, and logs every prediction to <code>nfl_data/predictions_log.csv</code>. Check the
calibration and the long-run table before trusting a market.</p>
<p><b>Limits.</b> The model doesn't know about injuries, suspensions, weather or rest. It also only sees closing
lines, which are the hardest to beat; the backtest is a conservative test. Check injury reports and the starting QB before
betting, and shop for the best price.</p>
<p class="sub">Settings live at the top of nfl_edge.py under CONFIG.</p>"""


def how_text(md):
    rep = {"POINTS_BLEND": f"{POINTS_BLEND * 100:.0f}", "HALF_LIFE": str(HALF_LIFE_WEEKS),
           "LAST_SEASON": f"{LAST_SEASON_WEIGHT:.0%}", "RIDGE": f"{RIDGE:g}", "TOTAL_SD": f"{md.total_sd:.0f}",
           "PHL": str(PLAYER_HALF_LIFE), "CUSH_S": f"{CUSHION_SIDES * 100:.1f}", "CUSH_P": f"{CUSHION_PROPS * 100:.0f}",
           "CUSH_E": f"{CUSHION_EARLY * 100:.1f}", "HIGH_PROB": f"{HIGH_PROB:.0%}", "MIN_PROB": f"{MIN_PROB:.0%}",
           "GAP": f"{MAX_MARKET_GAP * 100:.0f}"}
    t = HOW
    for k in sorted(rep, key=len, reverse=True):
        t = t.replace(k, rep[k])
    return t


JS = r"""
function amToDec(s){var a=parseFloat(String(s).replace(/[^0-9+\-.]/g,''));if(isNaN(a)||Math.abs(a)<100)return null;return a>0?1+a/100:1+100/Math.abs(a);}
function decToAm(d){return d>=2?'+'+Math.round((d-1)*100):String(-Math.round(100/(d-1)));}
document.querySelectorAll('nav button').forEach(function(b){b.addEventListener('click',function(){
 document.querySelectorAll('nav button').forEach(function(x){x.classList.remove('on')});
 document.querySelectorAll('.tab').forEach(function(x){x.classList.remove('on')});
 b.classList.add('on');document.getElementById(b.dataset.t).classList.add('on');});});
document.querySelectorAll('input.odds').forEach(function(inp){inp.addEventListener('input',function(){
 var tr=inp.closest('[data-p]'),out=tr.querySelector('.ev');var d=amToDec(inp.value);
 if(!d){out.textContent='';out.className='ev';return;}
 var p=+tr.dataset.p,lo=+tr.dataset.lo,ev=p*d-1,evlo=lo*d-1;
 var cls=evlo>0?'v':(ev>0?'l':'a'),word=evlo>0?'Value':(ev>0?'Lean':'Avoid');
 out.className='ev '+cls;out.textContent=(ev>=0?'+':'')+(ev*100).toFixed(1)+'% · '+word;});});
document.querySelectorAll('table.sortable th').forEach(function(th,i){th.classList.add('sort');th.addEventListener('click',function(){
 var tb=th.closest('table').tBodies[0],rows=Array.from(tb.rows),asc=th.dataset.asc!=='1';th.dataset.asc=asc?'1':'0';
 rows.sort(function(a,b){var x=a.cells[i].dataset.v,y=b.cells[i].dataset.v;x=x===undefined?a.cells[i].textContent:x;y=y===undefined?b.cells[i].textContent:y;
 var nx=parseFloat(x),ny=parseFloat(y);if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;return asc?String(x).localeCompare(y):String(y).localeCompare(x);});
 rows.forEach(function(r){tb.appendChild(r)});});});
document.querySelectorAll('input.odds').forEach(function(i){if(i.value)i.dispatchEvent(new Event('input'));});
function erf(x){var s=x<0?-1:1;x=Math.abs(x);var t=1/(1+0.3275911*x);var y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);return s*y;}
function ncdf(x){return 0.5*(1+erf(x/Math.SQRT2));}
function pmfP(l,k){var p=Math.exp(-l);for(var i=1;i<=k;i++)p*=l/i;return p;}
function atLeast(kind,m,sd,k){if(kind==='n')return 1-ncdf((k-0.5-m)/sd);var s=0;for(var i=0;i<k;i++)s+=pmfP(m,i);return Math.max(0,1-s);}
function cal(p){p=Math.min(Math.max(p,1e-4),1-1e-4);var k=CALK;return 1/(1+Math.exp(-k*Math.log(p/(1-p))));}
function exactly(kind,m,sd,k){if(kind==='n')return ncdf((k+0.5-m)/sd)-ncdf((k-0.5-m)/sd);return pmfP(m,k);}
document.querySelectorAll('div.calc').forEach(function(c){
 var sel=c.querySelector('select'),ln=c.querySelector('.ln'),po=c.querySelector('.po'),pu=c.querySelector('.pu'),out=c.querySelector('.calcout'),cush=+c.dataset.c;
 function upd(){var o=sel.options[sel.selectedIndex],k=o.dataset.k,m=+o.dataset.m,sd=+o.dataset.sd,L=parseFloat(ln.value);
  if(isNaN(L)){out.innerHTML='<span class="sub">Type a line to price it.</span>';return;}
  var over=atLeast(k,m,sd,Math.floor(L)+1),push=(L===Math.floor(L))?exactly(k,m,sd,L):0,under=Math.max(0,1-over-push);
  var po_=cal(over/(1-push)),pu_=1-po_;
  function side(name,p,priceTxt){p=Math.min(Math.max(p,0.01),0.99);var lo=Math.max(0.001,p-cush),s='<b>'+name+'</b> '+(p*100).toFixed(0)+'% · fair '+decToAm(1/p)+' · <span class="betat">bet at '+decToAm(1/lo)+'</span>';
   var d=amToDec(priceTxt);if(d){var ev=p*d-1,evlo=lo*d-1,cls=evlo>0?'v':(ev>0?'l':'a'),w=evlo>0?'Value':(ev>0?'Lean':'Avoid');s+=' <span class="ev '+cls+'">'+(ev>=0?'+':'')+(ev*100).toFixed(1)+'% · '+w+'</span>';}return s;}
  out.innerHTML=side('Over',po_,po.value)+'<br>'+side('Under',pu_,pu.value)+(push>0.01?'<br><span class="sub">push '+(push*100).toFixed(0)+'%</span>':'');}
 [sel,ln,po,pu].forEach(function(x){x.addEventListener('input',upd);x.addEventListener('change',upd);});upd();});
"""


def render_page(tabs, generated, games_in_model):
    on = ' class="on"'
    nav = "".join(f'<button{on if i == 0 else ""} data-t="{tid}">{label}</button>'
                  for i, (tid, label, _) in enumerate(tabs))
    secs = "".join(f'<section class="tab{" on" if i == 0 else ""}" id="{tid}">{body}</section>'
                   for i, (tid, _, body) in enumerate(tabs))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>NFL Edge Board</title>\n'
            f'<style>{CSS}</style></head><body>\n'
            f'<header><h1>NFL Edge Board</h1><div class="sub">Generated {generated} &middot; {SEASON} season &middot; '
            f'{games_in_model} games in model &middot; tap a game to open the full breakdown &middot; '
            f'<a class="backlink" href="{SITE_LINK}">The Stat Sheet &rarr;</a></div></header>\n'
            f'<nav><div class="in">{nav}</div></nav><main>{secs}</main>\n<script>{JS.replace("CALK", str(PROP_CALIB_K))}</script></body></html>')


# ==================================================================
# LOG + GITHUB
# ==================================================================
def write_log(weeks, path):
    rows = []
    for wk in weeks:
        for og in wk["games"]:
            g = og["g"]
            for mk in og["markets"]:
                rows.append({"season": wk["season"], "week": wk["week"], "game_id": g.game_id, "market": mk["market"],
                             "bet": mk["label"], "model_p": round(mk["p"], 4), "market_p": None if mk["mkt"] is None else round(mk["mkt"], 4),
                             "price": mk["price"], "fair": mk["fair"], "bet_at": mk["bet_at"],
                             "top_bet": mk.get("tag", "") if any(mk is m for _, m in wk.get("picks", [])) else "",
                             "outcome": mk.get("out"), "units": mk.get("pl") if any(mk is m for _, m in wk.get("picks", [])) else None})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"  [saved] {path} ({len(rows)} rows)")


def gh(method, path, **kwargs):
    r = requests.request(method, f"https://api.github.com/repos/{GITHUB_REPO}{path}",
                         headers={"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json",
                                  "X-GitHub-Api-Version": "2022-11-28"}, timeout=120, **kwargs)
    if r.status_code >= 300:
        raise RuntimeError(f"GitHub {method} {path} -> {r.status_code}: {r.text[:300]}")
    return r.json()


def push_to_github(files):
    print(f"\n=== Pushing to GitHub ({GITHUB_REPO}@{GITHUB_BRANCH}) ===")
    if not GITHUB_TOKEN:
        print("  [skip] GITHUB_TOKEN isn't set")
        return
    try:
        head = gh("GET", f"/git/ref/heads/{GITHUB_BRANCH}")["object"]["sha"]
        base_tree = gh("GET", f"/git/commits/{head}")["tree"]["sha"]
        existing = {t["path"]: t["sha"] for t in gh("GET", f"/git/trees/{base_tree}?recursive=1").get("tree", [])}
        tree = []
        for local in files:
            repo_path = local.replace(os.sep, "/").lstrip("./")
            data = open(local, "rb").read()
            if existing.get(repo_path) == hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest():
                print(f"  [same] {repo_path}")
                continue
            blob = gh("POST", "/git/blobs", json={"content": base64.b64encode(data).decode(), "encoding": "base64"})
            tree.append({"path": repo_path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
            print(f"  [upload] {repo_path}")
        if not tree:
            print("  Already up to date.")
            return
        new_tree = gh("POST", "/git/trees", json={"base_tree": base_tree, "tree": tree})["sha"]
        msg = f"Update NFL Edge Board ({time.strftime('%Y-%m-%d')})"
        commit = gh("POST", "/git/commits", json={"message": msg, "tree": new_tree, "parents": [head]})["sha"]
        gh("PATCH", f"/git/refs/heads/{GITHUB_BRANCH}", json={"sha": commit})
        print(f"  [done] {commit[:7]} \"{msg}\"")
    except RuntimeError as e:
        print(f"  [error] {e}")


# ==================================================================
# MAIN
# ==================================================================
def pick_target_week(games):
    g = games[(games.season == SEASON) & (games.game_type == "REG")]
    open_weeks = sorted(g[~g.done].week.unique())
    return int(open_weeks[0]) if open_weeks else int(g.week.max())


def main():
    t0 = time.time()
    print("=== Loading data ===")
    games = load_games()
    target = TARGET_WEEK or pick_target_week(games)
    seasons = sorted(set(range(min(LONG_RUN_SEASONS + [SEASON]) - SEASONS_BACK, SEASON + 1)))
    st = load_stats([s for s in seasons if s >= 2022])
    tg = team_game_table(games, st)
    tg = add_xpts(tg, fit_xpts(tg[tg.season < SEASON]))
    md = MarginDist(games, SEASON)
    print(f"  target: {SEASON} week {target}")

    print("=== Walk-forward weeks ===")
    weeks = []
    for w in range(FIRST_GRADED_WEEK, target + 1):
        wk = grade_week(build_week(SEASON, w, games, tg, st, md), st)
        weeks.append(wk)
        print(f"  week {w}: {len(wk['games'])} games, {len(wk['picks'])} top bets, {len(wk['props'])} players")
    tgt = weeks[-1]
    nxt = None
    ng = games[(games.season == SEASON) & (games.week == target + 1)]
    if len(ng) and ng.spread_line.notna().any() and not ng.done.any():
        nxt = grade_week(build_week(SEASON, target + 1, games, tg, st, md), st)
        print(f"  week {target + 1} (next): {len(nxt['games'])} games, {len(nxt['picks'])} top bets")

    long_run = {}
    if LONG_RUN_SEASONS:
        print("=== Long-run check ===")
        for s in LONG_RUN_SEASONS:
            mds = MarginDist(games, s)
            long_run[s] = [grade_week(build_week(s, w, games, tg, st, mds, with_props=False), st, with_props=False)
                           for w in range(2, 19)]
            sm = summarize_games(long_run[s])
            print(f"  {s}: top bets {wlp(sm['bets'])}, {sm['units']:+.1f}u")

    pending = sum(1 for og in tgt["games"] if not og["g"].done)
    done = len(tgt["games"]) - pending
    top_list = []
    if nxt:
        top_list.append((nxt, f"Week {nxt['week']} &middot; upcoming"))
    top_list.append((tgt, f"Week {target} &middot; " + (f"graded ({done} of {len(tgt['games'])} played)" if done else "upcoming")))
    tabs = [("top", "Top bets", render_top(top_list, long_run_note(long_run)))]
    sub = (f"Built with every game and stat before week {target}; finished games show the result. "
           f"{pending} game(s) still to play." if done else "Built with every game and stat so far.")
    tabs.append(("wk", f"Week {target}", render_week(tgt, games, f"Week {target}", sub)))
    if nxt:
        tabs.append(("nx", f"Week {target + 1}", render_week(
            nxt, games, f"Week {target + 1}",
            f"Preview built with everything through week {target}" +
            (" (unfinished week-" + str(target) + " games aren't in yet - rerun after they finish)." if pending else "."))))
    latest = nxt or tgt
    tabs.append(("lg", "League", render_league(games, tg, latest["ratings"], SEASON, latest["week"])))
    tabs.append(("tr", "Track record", render_track(tgt, weeks, long_run)))
    tabs.append(("hw", "How it works", '<div class="prose">' + how_text(md) + '</div>'))

    n_games = int(((tg.ord < SEASON * 25 + target) & (tg.season >= SEASON - SEASONS_BACK)).sum() / 2)
    page = render_page(tabs, datetime.now().strftime("%a %b %-d, %-I:%M %p"), n_games)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"  [saved] {OUTPUT_HTML} ({len(page) / 1e6:.1f} MB)")
    write_log(weeks + ([nxt] if nxt else []), LOG_CSV)
    if PUSH_TO_GITHUB:
        push_to_github([OUTPUT_HTML, LOG_CSV])
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
