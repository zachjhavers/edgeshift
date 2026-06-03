"""
Build rolling Four Factors + Elo features for every completed NBA game.
Processes games strictly chronologically so no future data leaks into features.

Per-game stats computed:
  eFG%   = (FGM + 0.5·FG3M) / FGA
  TOV%   = TOV / (FGA + 0.44·FTA + TOV)
  ORB%   = OREB / (OREB + opp_DREB)
  FT rate = FTA / FGA
  Poss   ≈ FGA − OREB + TOV + 0.44·FTA
  ORtg   = PTS / Poss × 100
  DRtg   = opp_PTS / opp_Poss × 100
  Net Rtg = ORtg − DRtg
  Pace   = (Poss + opp_Poss) / 2
"""

from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from db import get_conn, setup_db
from utils import ELO_DEFAULT, ELO_K, FEATURES, MIN_GAMES, ROLLING_WINDOW


def _poss(fga: float, oreb: float, tov: float, fta: float) -> float:
    return max(fga - oreb + tov + 0.44 * fta, 1.0)


def _game_stats(row: dict, side: str) -> dict:
    opp   = "away" if side == "home" else "home"
    fgm   = float(row.get(f"{side}_fgm") or 0)
    fga   = float(row.get(f"{side}_fga") or 1)
    fg3m  = float(row.get(f"{side}_fg3m") or 0)
    ftm   = float(row.get(f"{side}_ftm") or 0)
    fta   = float(row.get(f"{side}_fta") or 0)
    oreb  = float(row.get(f"{side}_oreb") or 0)
    dreb  = float(row.get(f"{side}_dreb") or 0)
    tov   = float(row.get(f"{side}_tov") or 0)
    pts   = float(row.get(f"{side}_pts") or 0)

    o_fga  = float(row.get(f"{opp}_fga") or 1)
    o_oreb = float(row.get(f"{opp}_oreb") or 0)
    o_dreb = float(row.get(f"{opp}_dreb") or 0)
    o_tov  = float(row.get(f"{opp}_tov") or 0)
    o_fta  = float(row.get(f"{opp}_fta") or 0)
    o_pts  = float(row.get(f"{opp}_pts") or 0)

    my_poss  = _poss(fga, oreb, tov, fta)
    opp_poss = _poss(o_fga, o_oreb, o_tov, o_fta)

    return {
        "efg":     (fgm + 0.5 * fg3m) / fga,
        "tov_pct": tov / (fga + 0.44 * fta + tov) if (fga + 0.44 * fta + tov) > 0 else 0.14,
        "orb_pct": oreb / (oreb + o_dreb) if (oreb + o_dreb) > 0 else 0.25,
        "ftr":     fta / fga,
        "ortg":    pts / my_poss * 100,
        "drtg":    o_pts / opp_poss * 100,
        "net_rtg": (pts / my_poss - o_pts / opp_poss) * 100,
        "pace":    (my_poss + opp_poss) / 2,
    }


def _avg(stats_list: list[dict]) -> dict:
    if not stats_list:
        return {}
    keys = stats_list[0].keys()
    return {k: sum(s[k] for s in stats_list) / len(stats_list) for k in keys}


def _elo_expected(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((away_elo - home_elo) / 400.0))


def build_all_features():
    """
    Rebuild features table from all completed games in the games table.
    Safe to re-run: uses INSERT OR REPLACE.
    """
    setup_db()
    conn = get_conn()

    rows = conn.execute("""
        SELECT game_id, game_date, season, game_type,
               home_team, away_team, home_win,
               home_pts, home_fgm, home_fga, home_fg3m, home_fg3a,
               home_ftm, home_fta, home_oreb, home_dreb, home_tov,
               away_pts, away_fgm, away_fga, away_fg3m, away_fg3a,
               away_ftm, away_fta, away_oreb, away_dreb, away_tov
        FROM games
        WHERE home_pts IS NOT NULL AND away_pts IS NOT NULL
        ORDER BY game_date ASC, game_id ASC
    """).fetchall()
    conn.close()

    print(f"Building features for {len(rows)} completed games...")

    elo: dict[str, float]            = defaultdict(lambda: ELO_DEFAULT)
    history: dict[str, list[dict]]   = defaultdict(list)
    last_date: dict[str, str]        = {}
    features: list[dict]             = []

    for row in rows:
        r         = dict(row)
        game_id   = r["game_id"]
        game_date = r["game_date"]
        home      = r["home_team"]
        away      = r["away_team"]
        home_win  = r["home_win"]

        h_hist = history[home][-ROLLING_WINDOW:]
        a_hist = history[away][-ROLLING_WINDOW:]

        if len(h_hist) >= MIN_GAMES and len(a_hist) >= MIN_GAMES:
            h_avg = _avg(h_hist)
            a_avg = _avg(a_hist)

            h_last     = last_date.get(home)
            a_last     = last_date.get(away)
            h_rest     = (datetime.strptime(game_date, "%Y-%m-%d") -
                          datetime.strptime(h_last, "%Y-%m-%d")).days if h_last else 3.0
            a_rest     = (datetime.strptime(game_date, "%Y-%m-%d") -
                          datetime.strptime(a_last, "%Y-%m-%d")).days if a_last else 3.0
            h_elo      = elo[home]
            a_elo      = elo[away]

            features.append({
                "game_id":       game_id,
                "game_date":     game_date,
                "season":        r["season"],
                "game_type":     r["game_type"],
                "home_team":     home,
                "away_team":     away,
                "home_efg":      h_avg["efg"],
                "home_tov_pct":  h_avg["tov_pct"],
                "home_orb_pct":  h_avg["orb_pct"],
                "home_ftr":      h_avg["ftr"],
                "home_ortg":     h_avg["ortg"],
                "home_drtg":     h_avg["drtg"],
                "home_net_rtg":  h_avg["net_rtg"],
                "home_pace":     h_avg["pace"],
                "away_efg":      a_avg["efg"],
                "away_tov_pct":  a_avg["tov_pct"],
                "away_orb_pct":  a_avg["orb_pct"],
                "away_ftr":      a_avg["ftr"],
                "away_ortg":     a_avg["ortg"],
                "away_drtg":     a_avg["drtg"],
                "away_net_rtg":  a_avg["net_rtg"],
                "away_pace":     a_avg["pace"],
                "efg_diff":      h_avg["efg"]     - a_avg["efg"],
                "tov_pct_diff":  a_avg["tov_pct"] - h_avg["tov_pct"],
                "orb_pct_diff":  h_avg["orb_pct"] - a_avg["orb_pct"],
                "ftr_diff":      h_avg["ftr"]     - a_avg["ftr"],
                "net_rtg_diff":  h_avg["net_rtg"] - a_avg["net_rtg"],
                "pace_avg":      (h_avg["pace"] + a_avg["pace"]) / 2,
                "home_rest_days": h_rest,
                "away_rest_days": a_rest,
                "rest_diff":     h_rest - a_rest,
                "home_elo":      h_elo,
                "away_elo":      a_elo,
                "elo_diff":      h_elo - a_elo,
                "home_win":      home_win,
            })

        # Update history with this game's stats
        h_stats = _game_stats(r, "home")
        a_stats = _game_stats(r, "away")
        history[home].append(h_stats)
        history[away].append(a_stats)
        last_date[home] = game_date
        last_date[away] = game_date

        # Update Elo after the game
        if home_win is not None:
            expected = _elo_expected(elo[home], elo[away])
            actual   = float(home_win)
            elo[home] += ELO_K * (actual - expected)
            elo[away] += ELO_K * ((1 - actual) - (1 - expected))

    # Write features to DB
    conn = get_conn()
    conn.execute("DELETE FROM features")
    conn.executemany("""
        INSERT OR REPLACE INTO features (
            game_id, game_date, season, game_type, home_team, away_team,
            home_efg, home_tov_pct, home_orb_pct, home_ftr,
            home_ortg, home_drtg, home_net_rtg, home_pace,
            away_efg, away_tov_pct, away_orb_pct, away_ftr,
            away_ortg, away_drtg, away_net_rtg, away_pace,
            efg_diff, tov_pct_diff, orb_pct_diff, ftr_diff,
            net_rtg_diff, pace_avg,
            home_rest_days, away_rest_days, rest_diff,
            home_elo, away_elo, elo_diff, home_win
        ) VALUES (
            :game_id, :game_date, :season, :game_type, :home_team, :away_team,
            :home_efg, :home_tov_pct, :home_orb_pct, :home_ftr,
            :home_ortg, :home_drtg, :home_net_rtg, :home_pace,
            :away_efg, :away_tov_pct, :away_orb_pct, :away_ftr,
            :away_ortg, :away_drtg, :away_net_rtg, :away_pace,
            :efg_diff, :tov_pct_diff, :orb_pct_diff, :ftr_diff,
            :net_rtg_diff, :pace_avg,
            :home_rest_days, :away_rest_days, :rest_diff,
            :home_elo, :away_elo, :elo_diff, :home_win
        )
    """, features)

    # Persist current Elo ratings for quick prediction-time lookup
    conn.execute("DELETE FROM team_elo")
    conn.executemany(
        "INSERT INTO team_elo (team, elo, last_updated) VALUES (?, ?, datetime('now'))",
        [(team, round(rating, 2)) for team, rating in elo.items()],
    )

    conn.commit()
    conn.close()
    print(f"  {len(features)} feature rows written. Elo updated for {len(elo)} teams.")


def get_prediction_features(
    home_team: str,
    away_team: str,
    game_date: str,
    conn,
) -> dict | None:
    """
    Compute pre-game features for a single upcoming game.
    Returns None if either team doesn't have enough prior games.
    """
    rows = conn.execute("""
        SELECT game_date, home_team, away_team,
               home_pts, home_fgm, home_fga, home_fg3m, home_fg3a,
               home_ftm, home_fta, home_oreb, home_dreb, home_tov,
               away_pts, away_fgm, away_fga, away_fg3m, away_fg3a,
               away_ftm, away_fta, away_oreb, away_dreb, away_tov
        FROM games
        WHERE game_date < ? AND home_pts IS NOT NULL
        ORDER BY game_date ASC
    """, (game_date,)).fetchall()

    h_hist: list[dict] = []
    a_hist: list[dict] = []
    h_last = a_last = None

    for row in rows:
        r = dict(row)
        gd = r["game_date"]
        if r["home_team"] == home_team:
            h_hist.append(_game_stats(r, "home"))
            h_last = gd
        elif r["away_team"] == home_team:
            h_hist.append(_game_stats(r, "away"))
            h_last = gd
        if r["home_team"] == away_team:
            a_hist.append(_game_stats(r, "home"))
            a_last = gd
        elif r["away_team"] == away_team:
            a_hist.append(_game_stats(r, "away"))
            a_last = gd

    h_hist = h_hist[-ROLLING_WINDOW:]
    a_hist = a_hist[-ROLLING_WINDOW:]

    if len(h_hist) < MIN_GAMES or len(a_hist) < MIN_GAMES:
        return None

    h_avg  = _avg(h_hist)
    a_avg  = _avg(a_hist)
    d_cur  = datetime.strptime(game_date, "%Y-%m-%d")
    h_rest = (d_cur - datetime.strptime(h_last, "%Y-%m-%d")).days if h_last else 3.0
    a_rest = (d_cur - datetime.strptime(a_last, "%Y-%m-%d")).days if a_last else 3.0

    elo_row_h = conn.execute("SELECT elo FROM team_elo WHERE team = ?", (home_team,)).fetchone()
    elo_row_a = conn.execute("SELECT elo FROM team_elo WHERE team = ?", (away_team,)).fetchone()
    h_elo = float(elo_row_h["elo"]) if elo_row_h else ELO_DEFAULT
    a_elo = float(elo_row_a["elo"]) if elo_row_a else ELO_DEFAULT

    return {
        "home_efg":      h_avg["efg"],
        "home_tov_pct":  h_avg["tov_pct"],
        "home_orb_pct":  h_avg["orb_pct"],
        "home_ftr":      h_avg["ftr"],
        "home_ortg":     h_avg["ortg"],
        "home_drtg":     h_avg["drtg"],
        "home_net_rtg":  h_avg["net_rtg"],
        "home_pace":     h_avg["pace"],
        "away_efg":      a_avg["efg"],
        "away_tov_pct":  a_avg["tov_pct"],
        "away_orb_pct":  a_avg["orb_pct"],
        "away_ftr":      a_avg["ftr"],
        "away_ortg":     a_avg["ortg"],
        "away_drtg":     a_avg["drtg"],
        "away_net_rtg":  a_avg["net_rtg"],
        "away_pace":     a_avg["pace"],
        "efg_diff":      h_avg["efg"]     - a_avg["efg"],
        "tov_pct_diff":  a_avg["tov_pct"] - h_avg["tov_pct"],
        "orb_pct_diff":  h_avg["orb_pct"] - a_avg["orb_pct"],
        "ftr_diff":      h_avg["ftr"]     - a_avg["ftr"],
        "net_rtg_diff":  h_avg["net_rtg"] - a_avg["net_rtg"],
        "pace_avg":      (h_avg["pace"] + a_avg["pace"]) / 2,
        "home_rest_days": h_rest,
        "away_rest_days": a_rest,
        "rest_diff":     h_rest - a_rest,
        "home_elo":      h_elo,
        "away_elo":      a_elo,
        "elo_diff":      h_elo - a_elo,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    build_all_features()
