"""
Inference-only: muat artefak model_binary.pkl dan prediksi peluang menang
untuk pasangan tim (home vs away).

Cara pakai:
    python model_predict.py --predict "Liverpool" "Man United"

Jika argumen tidak diberikan, akan diminta input interaktif.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

ARTIFACT_PATH = Path("analysis_outputs") / "model_binary.pkl"
DATA_PATH = Path("dataset/ALL_leagues_preprocessed.csv")


def load_artifact():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Artefak {ARTIFACT_PATH} tidak ditemukan. Jalankan dulu: python model_train.py"
        )
    obj = joblib.load(ARTIFACT_PATH)
    required_keys = {"model", "feature_cols", "home_stats", "away_stats"}
    if not required_keys.issubset(set(obj.keys())):
        raise ValueError(f"Artefak tidak lengkap, kunci yang ada: {obj.keys()}")
    return obj


def get_h2h_stats(df: pd.DataFrame, home_team: str, away_team: str) -> dict:
    """Hitung head-to-head stats dari dataset historis."""
    matches = df[
        (
            ((df["HomeTeam"] == home_team) & (df["AwayTeam"] == away_team))
            | ((df["HomeTeam"] == away_team) & (df["AwayTeam"] == home_team))
        )
    ].tail(5)

    if len(matches) == 0:
        return {"h2h_home_wins": 0.0, "h2h_away_wins": 0.0, "h2h_draws": 0.0, "h2h_matches": 0.0}

    h_wins = 0
    a_wins = 0
    draws = 0

    for _, m in matches.iterrows():
        if m["HomeTeam"] == home_team:
            if m["FTR"] == "H":
                h_wins += 1
            elif m["FTR"] == "A":
                a_wins += 1
            else:
                draws += 1
        else:
            if m["FTR"] == "A":
                h_wins += 1
            elif m["FTR"] == "H":
                a_wins += 1
            else:
                draws += 1

    n = len(matches)
    return {
        "h2h_home_wins": h_wins / n,
        "h2h_away_wins": a_wins / n,
        "h2h_draws": draws / n,
        "h2h_matches": float(n),
    }


def predict_match(artifact, home_team: str, away_team: str, df_hist: pd.DataFrame = None) -> pd.Series:
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    home_stats = artifact["home_stats"]
    away_stats = artifact["away_stats"]

    h_row = home_stats[home_stats["Team"].str.lower() == home_team.lower()]
    a_row = away_stats[away_stats["Team"].str.lower() == away_team.lower()]
    if h_row.empty:
        raise ValueError(f"Tidak ada histori cukup untuk tim kandang: {home_team}")
    if a_row.empty:
        raise ValueError(f"Tidak ada histori cukup untuk tim tandang: {away_team}")

    h_row = h_row.iloc[0]
    a_row = a_row.iloc[0]

    # Get values dengan fallback
    def safe_get(row, key, default=0.0):
        val = row.get(key)
        return val if val is not None and not pd.isna(val) else default

    h_avg_gf = safe_get(h_row, "home_avg_gf")
    h_avg_ga = safe_get(h_row, "home_avg_ga")
    h_avg_pts = safe_get(h_row, "home_avg_pts")
    h_win_rate = safe_get(h_row, "home_win_rate")
    h_avg_gf_long = safe_get(h_row, "home_avg_gf_long", h_avg_gf)
    h_avg_pts_long = safe_get(h_row, "home_avg_pts_long", h_avg_pts)
    h_momentum = safe_get(h_row, "home_momentum", h_avg_pts - h_avg_pts_long)

    a_avg_gf = safe_get(a_row, "away_avg_gf")
    a_avg_ga = safe_get(a_row, "away_avg_ga")
    a_avg_pts = safe_get(a_row, "away_avg_pts")
    a_win_rate = safe_get(a_row, "away_win_rate")
    a_avg_gf_long = safe_get(a_row, "away_avg_gf_long", a_avg_gf)
    a_avg_pts_long = safe_get(a_row, "away_avg_pts_long", a_avg_pts)
    a_momentum = safe_get(a_row, "away_momentum", a_avg_pts - a_avg_pts_long)

    # Head-to-head
    h2h = {"h2h_home_wins": 0.0, "h2h_away_wins": 0.0, "h2h_draws": 0.0, "h2h_matches": 0.0}
    if df_hist is not None:
        h2h = get_h2h_stats(df_hist, home_team, away_team)

    row = {
        "home_avg_gf": h_avg_gf,
        "home_avg_ga": h_avg_ga,
        "home_avg_pts": h_avg_pts,
        "home_win_rate": h_win_rate,
        "home_avg_gf_long": h_avg_gf_long,
        "home_avg_pts_long": h_avg_pts_long,
        "home_momentum": h_momentum,
        "away_avg_gf": a_avg_gf,
        "away_avg_ga": a_avg_ga,
        "away_avg_pts": a_avg_pts,
        "away_win_rate": a_win_rate,
        "away_avg_gf_long": a_avg_gf_long,
        "away_avg_pts_long": a_avg_pts_long,
        "away_momentum": a_momentum,
        "diff_avg_gf": h_avg_gf - a_avg_gf,
        "diff_avg_ga": h_avg_ga - a_avg_ga,
        "diff_avg_pts": h_avg_pts - a_avg_pts,
        "diff_win_rate": h_win_rate - a_win_rate,
        "diff_momentum": h_momentum - a_momentum,
        "h2h_home_wins": h2h["h2h_home_wins"],
        "h2h_away_wins": h2h["h2h_away_wins"],
        "h2h_draws": h2h["h2h_draws"],
        "h2h_matches": h2h["h2h_matches"],
        "diff_h2h": h2h["h2h_home_wins"] - h2h["h2h_away_wins"],
        "LeagueName": "Unknown",
    }

    # Shots on target (if available)
    if "home_avg_sot_for" in h_row and "away_avg_sot_for" in a_row:
        h_sot_for = safe_get(h_row, "home_avg_sot_for")
        h_sot_against = safe_get(h_row, "home_avg_sot_against")
        a_sot_for = safe_get(a_row, "away_avg_sot_for")
        a_sot_against = safe_get(a_row, "away_avg_sot_against")
        row["diff_sot_for"] = h_sot_for - a_sot_for
        row["diff_sot_against"] = h_sot_against - a_sot_against

    X = pd.DataFrame([row])[feature_cols + ["LeagueName"]]
    proba_arr = model.predict_proba(X)[0]
    proba = pd.Series(proba_arr, index=model.classes_, name=f"{home_team} vs {away_team}")
    proba = proba.reindex(["H", "A"]).fillna(0)
    total = proba.sum()
    if total > 0:
        proba = proba / total
    return proba


def main():
    parser = argparse.ArgumentParser(description="Inference-only for binary match prediction.")
    parser.add_argument(
        "--predict",
        nargs=2,
        metavar=("HOME_TEAM", "AWAY_TEAM"),
        help="Prediksi probabilitas menang (home vs away).",
    )
    args = parser.parse_args()

    artifact = load_artifact()

    # Load historical data untuk head-to-head
    df_hist = None
    if DATA_PATH.exists():
        df_hist = pd.read_csv(DATA_PATH)
        df_hist["Date"] = pd.to_datetime(df_hist["Date"])

    if args.predict:
        home_team, away_team = args.predict
    else:
        home_team = input("Tim home: ").strip()
        away_team = input("Tim away: ").strip()

    proba = predict_match(artifact, home_team, away_team, df_hist)
    print(f"\nProbabilitas menang {home_team} vs {away_team}:")
    print(f"  {home_team} (H): {proba.get('H', 0)*100:.2f}%")
    print(f"  {away_team} (A): {proba.get('A', 0)*100:.2f}%")


if __name__ == "__main__":
    main()

