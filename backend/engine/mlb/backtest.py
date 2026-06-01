"""
Walk-forward backtest for the MLB EV model.

For each game in the test window, team stats are computed using only games
played BEFORE that date — no look-ahead bias. The model is trained on the
training window only, then frozen for the entire test window.

P&L is simulated at -110/-110 (decimal 1.909) since historical odds are not
stored for all games. Real sportsbook lines will vary, so treat the P&L as directional.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss
from db import get_engine
from utils import FEATURES, PARK_FACTORS

STAKE          = 100
SIM_ODDS       = 1.909
BREAKEVEN      = 1 / SIM_ODDS
TEAM_WINDOW    = 15
STARTER_WINDOW = 5
BULLPEN_WINDOW = 10
ELO_K          = 20
ELO_HOME_ADV   = 35
ELO_INIT       = 1500
ELO_REGRESS    = 0.75

# ── helpers ──────────────────────────────────────────────────────────────────

def compute_elo_ratings(df: pd.DataFrame) -> dict:
    elo      = {}
    lookup   = {}
    prev_year = None

    for _, game in df.sort_values('game_date').iterrows():
        year = game['game_date'].year
        if prev_year is not None and year != prev_year:
            for team in list(elo.keys()):
                elo[team] = elo[team] * ELO_REGRESS + ELO_INIT * (1 - ELO_REGRESS)
        prev_year = year

        home, away = game['home_team'], game['away_team']
        elo_h = elo.get(home, ELO_INIT)
        elo_a = elo.get(away, ELO_INIT)

        e_home = 1 / (1 + 10 ** ((elo_a - elo_h - ELO_HOME_ADV) / 400))
        lookup[game['game_pk']] = {'home_elo_prob': e_home, 'elo_diff': elo_h - elo_a}

        actual = game['home_win']
        elo[home] = elo_h + ELO_K * (actual - e_home)
        elo[away] = elo_a + ELO_K * ((1 - actual) - (1 - e_home))

    return lookup


def get_team_stats(team: str, prior: pd.DataFrame) -> dict:
    as_home = prior[prior['home_team'] == team][
        ['game_date', 'home_pitch_velo', 'home_bat_exit_velo', 'home_xwoba']
    ].rename(columns={
        'home_pitch_velo': 'pitch_velo',
        'home_bat_exit_velo': 'bat_exit_velo',
        'home_xwoba': 'xwoba',
    })
    as_away = prior[prior['away_team'] == team][
        ['game_date', 'away_pitch_velo', 'away_bat_exit_velo', 'away_xwoba']
    ].rename(columns={
        'away_pitch_velo': 'pitch_velo',
        'away_bat_exit_velo': 'bat_exit_velo',
        'away_xwoba': 'xwoba',
    })
    combined = pd.concat([as_home, as_away]).sort_values('game_date').tail(TEAM_WINDOW)
    return combined.mean().to_dict()


def get_team_record(team: str, prior: pd.DataFrame) -> dict | None:
    home_games = prior[prior['home_team'] == team][
        ['game_date', 'home_win', 'final_home_score', 'final_away_score']
    ].copy()
    home_games['won']      = home_games['home_win'].astype(float)
    home_games['run_diff'] = home_games['final_home_score'] - home_games['final_away_score']

    away_games = prior[prior['away_team'] == team][
        ['game_date', 'home_win', 'final_home_score', 'final_away_score']
    ].copy()
    away_games['won']      = (1 - away_games['home_win']).astype(float)
    away_games['run_diff'] = away_games['final_away_score'] - away_games['final_home_score']

    combined = pd.concat([
        home_games[['game_date', 'won', 'run_diff']],
        away_games[['game_date', 'won', 'run_diff']],
    ]).sort_values('game_date').tail(TEAM_WINDOW)
    if len(combined) < 5:
        return None
    return {
        'win_pct':  combined['won'].mean(),
        'run_diff': combined['run_diff'].mean(),
    }


def get_starter_rolling_stats(pitcher_id, game_date: pd.Timestamp,
                               df_starters: pd.DataFrame) -> dict | None:
    prior = df_starters[
        (df_starters['pitcher'] == pitcher_id) &
        (df_starters['game_date'] < game_date)
    ].tail(STARTER_WINDOW)
    if len(prior) < 2:
        return None
    k  = prior['k_pct'].mean()
    bb = prior['bb_pct'].mean()
    return {
        'velo':            prior['avg_velo'].mean(),
        'k_pct':           k,
        'bb_pct':          bb,
        'k_minus_bb_pct':  k - bb,
        'xwoba_against':   prior['xwoba_against'].mean(),
        'ip':              prior['ip'].mean(),
    }


def get_rest_days(team: str, game_date: pd.Timestamp, df_games: pd.DataFrame) -> float:
    prior = df_games[
        ((df_games['home_team'] == team) | (df_games['away_team'] == team)) &
        (df_games['game_date'] < game_date)
    ]
    if prior.empty:
        return 3.0
    return float(min((game_date - prior['game_date'].max()).days, 7))


def get_bullpen_k_pct(team: str, game_date: pd.Timestamp,
                      df_bullpen_agg: pd.DataFrame) -> float | None:
    prior = df_bullpen_agg[
        (df_bullpen_agg['team'] == team) &
        (df_bullpen_agg['game_date'] < game_date)
    ].tail(BULLPEN_WINDOW)
    if len(prior) < 3:
        return None
    return float(prior['bp_k_pct'].mean())


def pnl_line(label: str, bets: pd.DataFrame) -> str:
    if bets.empty:
        return f"  {label}: no bets"
    wins      = bets['won'].sum()
    total     = len(bets)
    total_pnl = bets['pnl'].sum()
    roi       = total_pnl / (total * STAKE) * 100
    return (f"  {label}: {total} bets  |  {wins/total*100:.1f}% win rate  |  "
            f"P&L ${total_pnl:+,.0f}  |  ROI {roi:+.1f}%")

# ── main ─────────────────────────────────────────────────────────────────────

def run_backtest():
    print("=== MLB EV Walk-Forward Backtest ===\n")
    engine = get_engine()

    query = """
        SELECT
            game_pk,
            CAST(game_date AS DATE)                                                     AS game_date,
            home_team, away_team,
            MAX(home_score)                                                             AS final_home_score,
            MAX(away_score)                                                             AS final_away_score,
            AVG(CASE WHEN inning_topbot = 'Top' THEN release_speed             END)    AS home_pitch_velo,
            AVG(CASE WHEN inning_topbot = 'Bot' THEN release_speed             END)    AS away_pitch_velo,
            AVG(CASE WHEN inning_topbot = 'Bot' THEN launch_speed              END)    AS home_bat_exit_velo,
            AVG(CASE WHEN inning_topbot = 'Top' THEN launch_speed              END)    AS away_bat_exit_velo,
            AVG(CASE WHEN inning_topbot = 'Bot' THEN estimated_woba_using_speedangle END) AS home_xwoba,
            AVG(CASE WHEN inning_topbot = 'Top' THEN estimated_woba_using_speedangle END) AS away_xwoba
        FROM statcast_raw
        WHERE game_type = 'R'
        GROUP BY game_pk, CAST(game_date AS DATE), home_team, away_team
        HAVING MAX(home_score) IS NOT NULL AND MAX(away_score) IS NOT NULL
        ORDER BY game_date
    """
    df = pd.read_sql(query, engine, parse_dates=['game_date'])
    base_cols = ['home_pitch_velo', 'away_pitch_velo', 'home_bat_exit_velo',
                 'away_bat_exit_velo', 'home_xwoba', 'away_xwoba']
    df = df.dropna(subset=base_cols)
    df['home_win'] = (df['final_home_score'] > df['final_away_score']).astype(int)

    print("Computing Elo ratings...")
    elo_lookup = compute_elo_ratings(df)

    pitcher_query = """
        SELECT
            game_pk,
            CAST(game_date AS DATE)  AS game_date,
            home_team, away_team,
            pitcher,
            inning_topbot,
            AVG(release_speed)       AS avg_velo,
            SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END)::float /
                NULLIF(SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END), 0) AS k_pct,
            SUM(CASE WHEN events IN ('walk', 'intent_walk') THEN 1 ELSE 0 END)::float /
                NULLIF(SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END), 0) AS bb_pct,
            AVG(estimated_woba_using_speedangle) AS xwoba_against,
            COUNT(*)                 AS pitch_count,
            SUM(CASE
                WHEN events IN ('strikeout', 'field_out', 'force_out', 'sac_bunt', 'sac_fly',
                                'fielders_choice_out', 'other_out', 'caught_stealing_2b',
                                'caught_stealing_3b', 'caught_stealing_home',
                                'pickoff_caught_stealing_2b', 'pickoff_caught_stealing_3b',
                                'pickoff_caught_stealing_home', 'sac_bunt_double_play') THEN 1
                WHEN events IN ('grounded_into_double_play', 'strikeout_double_play',
                                'double_play', 'sac_fly_double_play') THEN 2
                WHEN events = 'triple_play' THEN 3
                ELSE 0 END) * 1.0 / 3.0 AS ip
        FROM statcast_raw
        WHERE game_type = 'R'
        GROUP BY game_pk, CAST(game_date AS DATE), home_team, away_team, pitcher, inning_topbot
    """
    df_pitcher  = pd.read_sql(pitcher_query, engine, parse_dates=['game_date'])
    starter_idx = df_pitcher.groupby(['game_pk', 'inning_topbot'])['pitch_count'].idxmax()
    df_pitcher['is_starter'] = False
    df_pitcher.loc[starter_idx, 'is_starter'] = True
    df_starters = df_pitcher.loc[starter_idx].reset_index(drop=True)
    starter_lookup = {(r['game_pk'], r['inning_topbot']): r['pitcher']
                      for _, r in df_starters.iterrows()}

    df_bp = df_pitcher[~df_pitcher['is_starter']].copy()
    # Top of inning → home team pitching. Bot of inning → away team pitching.
    df_bp['team'] = df_bp.apply(
        lambda r: r['home_team'] if r['inning_topbot'] == 'Top' else r['away_team'], axis=1
    )
    df_bullpen_agg = (
        df_bp.groupby(['team', 'game_date'])
        .agg(bp_k_pct=('k_pct', 'mean'))
        .reset_index()
        .sort_values(['team', 'game_date'])
    )

    # ── Walk-forward feature build (all games) ───────────────────────────────
    print("Building rolling features for all games (this may take a moment)...")
    all_rows = []
    for _, game in df.iterrows():
        prior = df[df['game_date'] < game['game_date']]
        if len(prior) < TEAM_WINDOW:
            continue

        team_h = get_team_stats(game['home_team'], prior)
        team_a = get_team_stats(game['away_team'], prior)
        if any(pd.isna(v) for v in {**team_h, **team_a}.values()):
            continue

        rec_h = get_team_record(game['home_team'], prior)
        rec_a = get_team_record(game['away_team'], prior)
        if rec_h is None or rec_a is None:
            continue

        home_starter_id = starter_lookup.get((game['game_pk'], 'Top'))
        away_starter_id = starter_lookup.get((game['game_pk'], 'Bot'))
        if home_starter_id is None or away_starter_id is None:
            continue

        sp_h = get_starter_rolling_stats(home_starter_id, game['game_date'], df_starters)
        sp_a = get_starter_rolling_stats(away_starter_id, game['game_date'], df_starters)
        if sp_h is None or sp_a is None:
            continue

        bp_h = get_bullpen_k_pct(game['home_team'], game['game_date'], df_bullpen_agg)
        bp_a = get_bullpen_k_pct(game['away_team'], game['game_date'], df_bullpen_agg)
        if bp_h is None or bp_a is None:
            continue

        elo    = elo_lookup.get(game['game_pk'], {'home_elo_prob': 0.5, 'elo_diff': 0.0})
        rest_h = get_rest_days(game['home_team'], game['game_date'], df)
        rest_a = get_rest_days(game['away_team'], game['game_date'], df)
        park_f = PARK_FACTORS.get(game['home_team'], 1.0)

        all_rows.append({
            'game_date':                   game['game_date'],
            'home_team':                   game['home_team'],
            'away_team':                   game['away_team'],
            'home_win':                    game['home_win'],
            'home_pitch_velo':             team_h['pitch_velo'],
            'away_pitch_velo':             team_a['pitch_velo'],
            'home_bat_exit_velo':          team_h['bat_exit_velo'],
            'away_bat_exit_velo':          team_a['bat_exit_velo'],
            'home_xwoba':                  team_h['xwoba'],
            'away_xwoba':                  team_a['xwoba'],
            'pitch_velo_diff':             team_h['pitch_velo']    - team_a['pitch_velo'],
            'bat_exit_velo_diff':          team_h['bat_exit_velo'] - team_a['bat_exit_velo'],
            'xwoba_diff':                  team_h['xwoba']         - team_a['xwoba'],
            'home_starter_velo':           sp_h['velo'],
            'away_starter_velo':           sp_a['velo'],
            'home_starter_k_pct':          sp_h['k_pct'],
            'away_starter_k_pct':          sp_a['k_pct'],
            'home_starter_bb_pct':         sp_h['bb_pct'],
            'away_starter_bb_pct':         sp_a['bb_pct'],
            'home_starter_k_minus_bb_pct': sp_h['k_minus_bb_pct'],
            'away_starter_k_minus_bb_pct': sp_a['k_minus_bb_pct'],
            'home_starter_xwoba_against':  sp_h['xwoba_against'],
            'away_starter_xwoba_against':  sp_a['xwoba_against'],
            'home_starter_ip':             sp_h['ip'],
            'away_starter_ip':             sp_a['ip'],
            'starter_velo_diff':           sp_h['velo']           - sp_a['velo'],
            'starter_k_pct_diff':          sp_h['k_pct']          - sp_a['k_pct'],
            'starter_bb_pct_diff':         sp_h['bb_pct']         - sp_a['bb_pct'],
            'starter_k_minus_bb_pct_diff': sp_h['k_minus_bb_pct'] - sp_a['k_minus_bb_pct'],
            'starter_xwoba_diff':          sp_h['xwoba_against']  - sp_a['xwoba_against'],
            'starter_ip_diff':             sp_h['ip']             - sp_a['ip'],
            'home_elo_prob':               elo['home_elo_prob'],
            'elo_diff':                    elo['elo_diff'],
            'home_rest_days':              rest_h,
            'away_rest_days':              rest_a,
            'rest_days_diff':              rest_h - rest_a,
            'home_park_factor':            park_f,
            'home_bullpen_k_pct':          bp_h,
            'away_bullpen_k_pct':          bp_a,
            'bullpen_k_pct_diff':          bp_h - bp_a,
            'home_win_pct_l15':            rec_h['win_pct'],
            'away_win_pct_l15':            rec_a['win_pct'],
            'home_run_diff_l15':           rec_h['run_diff'],
            'away_run_diff_l15':           rec_a['run_diff'],
            'win_pct_diff':                rec_h['win_pct']  - rec_a['win_pct'],
            'run_diff_diff':               rec_h['run_diff'] - rec_a['run_diff'],
        })

    df_feat = pd.DataFrame(all_rows)

    # ── Time-based 60/15/25 split: train / calibration / test ────────────────
    n          = len(df_feat)
    train_end  = int(n * 0.60)
    cal_end    = int(n * 0.75)

    df_train = df_feat.iloc[:train_end].copy()
    df_cal   = df_feat.iloc[train_end:cal_end].copy()
    df_test  = df_feat.iloc[cal_end:].copy()

    print(f"Train: {df_train['game_date'].min().date()} → {df_train['game_date'].max().date()} "
          f"({len(df_train)} games)")
    print(f"Cal:   {df_cal['game_date'].min().date()} → {df_cal['game_date'].max().date()} "
          f"({len(df_cal)} games)")
    print(f"Test:  {df_test['game_date'].min().date()} → {df_test['game_date'].max().date()} "
          f"({len(df_test)} games)\n")

    model_raw = XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=8,
        gamma=0.2,
        reg_lambda=2.0,
        reg_alpha=0.5,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
    )
    model_raw.fit(df_train[FEATURES], df_train['home_win'])

    cal_probs = model_raw.predict_proba(df_cal[FEATURES])[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(cal_probs, df_cal['home_win'].values)

    def predict_prob(X):
        return iso.transform(model_raw.predict_proba(X)[:, 1])

    # ── Test predictions ─────────────────────────────────────────────────────
    rows = []
    for _, game in df_test.iterrows():
        prob_home = predict_prob(game[FEATURES].to_frame().T.astype(float))[0]
        rows.append({
            'game_date':       game['game_date'],
            'home_team':       game['home_team'],
            'away_team':       game['away_team'],
            'prob_home':       prob_home,
            'prob_away':       1 - prob_home,
            'actual_home_win': game['home_win'],
        })

    results = pd.DataFrame(rows)
    print(f"Predictions generated: {len(results)}\n")

    # ── Accuracy metrics ─────────────────────────────────────────────────────
    y_true = results['actual_home_win']
    y_pred = (results['prob_home'] >= 0.5).astype(int)
    proba  = results['prob_home']

    accuracy = accuracy_score(y_true, y_pred)
    brier    = brier_score_loss(y_true, proba)
    baseline = y_true.mean()

    print("=== Accuracy ===")
    print(f"  Model accuracy:       {accuracy*100:.1f}%")
    print(f"  Always-home baseline: {baseline*100:.1f}%")
    print(f"  Brier score:          {brier:.4f}  (0.25 = random)")
    print(f"  Mean predicted prob:  {proba.mean():.3f}  (actual home win rate: {baseline:.3f})\n")

    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=5, strategy='quantile')
    print("=== Calibration curve ===")
    for pred, actual in zip(mean_pred, frac_pos):
        bar = '#' * int(actual * 20)
        print(f"  {pred*100:4.1f}% predicted  →  {actual*100:4.1f}% actual  {bar}")
    print()

    # ── Simulated P&L ────────────────────────────────────────────────────────
    home_bets = results[results['prob_home'] > BREAKEVEN].copy()
    home_bets['won'] = home_bets['actual_home_win'] == 1
    home_bets['pnl'] = np.where(home_bets['won'], STAKE * (SIM_ODDS - 1), -STAKE)

    away_bets = results[results['prob_away'] > BREAKEVEN].copy()
    away_bets['won'] = away_bets['actual_home_win'] == 0
    away_bets['pnl'] = np.where(away_bets['won'], STAKE * (SIM_ODDS - 1), -STAKE)

    all_bets = pd.concat([home_bets, away_bets]).sort_values('game_date').reset_index(drop=True)

    print(f"=== Simulated P&L (${STAKE}/bet at -110, threshold >{BREAKEVEN*100:.1f}%) ===")
    print(pnl_line("Home bets", home_bets))
    print(pnl_line("Away bets", away_bets))
    print(pnl_line("Combined ", all_bets))

    if not all_bets.empty:
        cum  = all_bets['pnl'].cumsum().values
        n    = len(cum)
        step = max(1, n // 10)
        print(f"\n  Cumulative P&L (${STAKE}/bet):")
        for i in range(0, n, step):
            bar_len = int(abs(cum[i]) / STAKE)
            bar = ('>' if cum[i] >= 0 else '<') * min(bar_len, 30)
            print(f"    Bet {i+1:3d}: ${cum[i]:+8,.0f}  {bar}")
        print(f"    Final:   ${cum[-1]:+8,.0f}")

    print(f"\nNote: P&L uses simulated -110 odds. Actual lines will vary.")


if __name__ == "__main__":
    run_backtest()
