"""
Train model binary (home vs away) dan simpan artefak untuk inference.

Keluaran:
- analysis_outputs/model_binary.pkl  (berisi model, feature_cols, home_stats, away_stats)
- analysis_outputs/predictions_binary.csv (prediksi pada test set)

Cara pakai:
    python model_train.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path("dataset/ALL_leagues_preprocessed.csv")
TEST_SPLIT_DATE = pd.Timestamp("2022-01-01")
ARTIFACT_PATH = Path("analysis_outputs") / "model_binary.pkl"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Tidak menemukan dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    if "Date" not in df.columns or "FTR" not in df.columns:
        raise ValueError("Dataset harus punya kolom Date dan FTR.")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def add_rolling_features(df: pd.DataFrame):
    """
    Enhanced rolling features:
    - Window 5 (short-term) + Window 10 (long-term form)
    - Head-to-head history
    - Momentum/trend indicators
    """
    work = df.copy()

    def pts(series_ftr: pd.Series, is_home: bool) -> pd.Series:
        if is_home:
            return series_ftr.map({"H": 3, "D": 1, "A": 0})
        return series_ftr.map({"A": 3, "D": 1, "H": 0})

    work["home_pts"] = pts(work["FTR"], True)
    work["away_pts"] = pts(work["FTR"], False)

    # === SHORT-TERM FORM (Window 5) ===
    work["home_avg_gf"] = (
        work.groupby("HomeTeam")["FTHG"]
        .transform(lambda s: s.shift().rolling(5, min_periods=3).mean())
    )
    work["home_avg_ga"] = (
        work.groupby("HomeTeam")["FTAG"]
        .transform(lambda s: s.shift().rolling(5, min_periods=3).mean())
    )
    work["home_avg_pts"] = (
        work.groupby("HomeTeam")["home_pts"]
        .transform(lambda s: s.shift().rolling(5, min_periods=3).mean())
    )
    work["home_win_rate"] = (
        work.groupby("HomeTeam")["FTR"]
        .transform(
            lambda s: s.shift()
            .map({"H": 1.0, "D": 0.0, "A": 0.0})
            .rolling(5, min_periods=3)
            .mean()
        )
    )

    work["away_avg_gf"] = (
        work.groupby("AwayTeam")["FTAG"]
        .transform(lambda s: s.shift().rolling(5, min_periods=3).mean())
    )
    work["away_avg_ga"] = (
        work.groupby("AwayTeam")["FTHG"]
        .transform(lambda s: s.shift().rolling(5, min_periods=3).mean())
    )
    work["away_avg_pts"] = (
        work.groupby("AwayTeam")["away_pts"]
        .transform(lambda s: s.shift().rolling(5, min_periods=3).mean())
    )
    work["away_win_rate"] = (
        work.groupby("AwayTeam")["FTR"]
        .transform(
            lambda s: s.shift()
            .map({"A": 1.0, "D": 0.0, "H": 0.0})
            .rolling(5, min_periods=3)
            .mean()
        )
    )

    # === LONG-TERM FORM (Window 10) ===
    work["home_avg_gf_long"] = (
        work.groupby("HomeTeam")["FTHG"]
        .transform(lambda s: s.shift().rolling(10, min_periods=5).mean())
    )
    work["home_avg_pts_long"] = (
        work.groupby("HomeTeam")["home_pts"]
        .transform(lambda s: s.shift().rolling(10, min_periods=5).mean())
    )
    work["away_avg_gf_long"] = (
        work.groupby("AwayTeam")["FTAG"]
        .transform(lambda s: s.shift().rolling(10, min_periods=5).mean())
    )
    work["away_avg_pts_long"] = (
        work.groupby("AwayTeam")["away_pts"]
        .transform(lambda s: s.shift().rolling(10, min_periods=5).mean())
    )

    # === MOMENTUM/TREND (recent vs older performance) ===
    work["home_momentum"] = work["home_avg_pts"] - work["home_avg_pts_long"]
    work["away_momentum"] = work["away_avg_pts"] - work["away_avg_pts_long"]

    # === HEAD-TO-HEAD HISTORY (optimized version) ===
    print("Menghitung head-to-head history...")
    work["h2h_home_wins"] = 0.0
    work["h2h_away_wins"] = 0.0
    work["h2h_draws"] = 0.0
    work["h2h_matches"] = 0.0

    # Buat key untuk pairing tim (sorted untuk konsistensi)
    work["team_pair"] = work.apply(
        lambda row: tuple(sorted([row["HomeTeam"], row["AwayTeam"]])), axis=1
    )

    # Group by team pair dan date untuk efisiensi
    for team_pair in work["team_pair"].unique():
        pair_matches = work[work["team_pair"] == team_pair].sort_values("Date").copy()

        for idx in pair_matches.index:
            current_match = pair_matches.loc[idx]
            current_date = current_match["Date"]
            home_team = current_match["HomeTeam"]
            away_team = current_match["AwayTeam"]

            # Ambil pertemuan sebelumnya (max 5)
            past = pair_matches[pair_matches["Date"] < current_date].tail(5)

            if len(past) > 0:
                h_wins = 0
                a_wins = 0
                draws = 0

                for _, pm in past.iterrows():
                    if pm["HomeTeam"] == home_team:
                        if pm["FTR"] == "H":
                            h_wins += 1
                        elif pm["FTR"] == "A":
                            a_wins += 1
                        else:
                            draws += 1
                    else:
                        if pm["FTR"] == "A":
                            h_wins += 1
                        elif pm["FTR"] == "H":
                            a_wins += 1
                        else:
                            draws += 1

                n = len(past)
                work.loc[idx, "h2h_home_wins"] = h_wins / n
                work.loc[idx, "h2h_away_wins"] = a_wins / n
                work.loc[idx, "h2h_draws"] = draws / n
                work.loc[idx, "h2h_matches"] = n

    # Drop kolom helper
    work = work.drop(columns=["team_pair"])

    # === SHOTS ON TARGET (if available) ===
    if "HST" in work.columns and "AST" in work.columns:
        work["home_avg_sot_for"] = (
            work.groupby("HomeTeam")["HST"]
            .transform(lambda s: s.shift().rolling(5, min_periods=3).mean())
        )
        work["home_avg_sot_against"] = (
            work.groupby("HomeTeam")["AST"]
            .transform(lambda s: s.shift().rolling(5, min_periods=3).mean())
        )
        work["away_avg_sot_for"] = (
            work.groupby("AwayTeam")["AST"]
            .transform(lambda s: s.shift().rolling(5, min_periods=3).mean())
        )
        work["away_avg_sot_against"] = (
            work.groupby("AwayTeam")["HST"]
            .transform(lambda s: s.shift().rolling(5, min_periods=3).mean())
        )

    # === DIFFERENCE FEATURES ===
    work["diff_avg_gf"] = work["home_avg_gf"] - work["away_avg_gf"]
    work["diff_avg_ga"] = work["home_avg_ga"] - work["away_avg_ga"]
    work["diff_avg_pts"] = work["home_avg_pts"] - work["away_avg_pts"]
    work["diff_win_rate"] = work["home_win_rate"] - work["away_win_rate"]
    work["diff_momentum"] = work["home_momentum"] - work["away_momentum"]
    work["diff_h2h"] = work["h2h_home_wins"] - work["h2h_away_wins"]

    feature_cols = [
        "home_avg_gf",
        "home_avg_ga",
        "home_avg_pts",
        "home_win_rate",
        "home_avg_gf_long",
        "home_avg_pts_long",
        "home_momentum",
        "away_avg_gf",
        "away_avg_ga",
        "away_avg_pts",
        "away_win_rate",
        "away_avg_gf_long",
        "away_avg_pts_long",
        "away_momentum",
        "diff_avg_gf",
        "diff_avg_ga",
        "diff_avg_pts",
        "diff_win_rate",
        "diff_momentum",
        "h2h_home_wins",
        "h2h_away_wins",
        "h2h_draws",
        "h2h_matches",
        "diff_h2h",
    ]

    if "home_avg_sot_for" in work.columns:
        work["diff_sot_for"] = work["home_avg_sot_for"] - work["away_avg_sot_for"]
        work["diff_sot_against"] = (
            work["home_avg_sot_against"] - work["away_avg_sot_against"]
        )
        feature_cols.extend(["diff_sot_for", "diff_sot_against"])

    work = work.dropna(subset=feature_cols)
    return work, feature_cols


def class_weights(y: pd.Series) -> dict:
    """Calculate class weights for imbalanced data."""
    counts = y.value_counts()
    n = len(y)
    k = counts.shape[0]
    return {cls: n / (k * cnt) for cls, cnt in counts.items()}


def build_pipeline(num_cols: list[str], cat_cols: list[str]):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    model = HistGradientBoostingClassifier(
        max_depth=10,
        learning_rate=0.06,
        max_iter=500,
        min_samples_leaf=10,
        l2_regularization=0.01,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    return Pipeline(steps=[("prep", preprocessor), ("model", model)])


def build_team_snapshots(df_feat: pd.DataFrame):
    home_cols = [
        "HomeTeam",
        "home_avg_gf",
        "home_avg_ga",
        "home_avg_pts",
        "home_win_rate",
        "home_avg_gf_long",
        "home_avg_pts_long",
        "home_momentum",
    ]
    if "home_avg_sot_for" in df_feat.columns:
        home_cols.extend(["home_avg_sot_for", "home_avg_sot_against"])

    away_cols = [
        "AwayTeam",
        "away_avg_gf",
        "away_avg_ga",
        "away_avg_pts",
        "away_win_rate",
        "away_avg_gf_long",
        "away_avg_pts_long",
        "away_momentum",
    ]
    if "away_avg_sot_for" in df_feat.columns:
        away_cols.extend(["away_avg_sot_for", "away_avg_sot_against"])
    home_stats = (
        df_feat.dropna(subset=["home_avg_gf", "home_avg_ga"])
        .groupby("HomeTeam")
        .tail(1)[home_cols]
        .rename(columns={"HomeTeam": "Team"})
        .reset_index(drop=True)
    )
    away_stats = (
        df_feat.dropna(subset=["away_avg_gf", "away_avg_ga"])
        .groupby("AwayTeam")
        .tail(1)[away_cols]
        .rename(columns={"AwayTeam": "Team"})
        .reset_index(drop=True)
    )
    return home_stats, away_stats


def train():
    df = load_data()
    df_feat, feature_cols = add_rolling_features(df)

    # binary only (drop draw)
    train_df = df_feat[df_feat["Date"] < TEST_SPLIT_DATE]
    test_df = df_feat[df_feat["Date"] >= TEST_SPLIT_DATE]
    train_df = train_df[train_df["FTR"].isin(["H", "A"])]
    test_df = test_df[test_df["FTR"].isin(["H", "A"])]

    X_train = train_df[feature_cols + ["LeagueName"]]
    y_train_raw = train_df["FTR"]
    X_test = test_df[feature_cols + ["LeagueName"]]
    y_test_raw = test_df["FTR"]
    
    y_train = y_train_raw
    y_test = y_test_raw

    pipe = build_pipeline(feature_cols, ["LeagueName"])
    cw = class_weights(y_train)
    sample_weight = y_train.map(cw)
    pipe.fit(X_train, y_train, model__sample_weight=sample_weight)

    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    ll = log_loss(y_test, proba)

    print("=== HGB Binary FTR (H vs A) ===")
    print(f"Train size: {len(train_df):,}  | Test size: {len(test_df):,}")
    print(f"Accuracy : {acc:.3f}")
    print(f"F1-macro : {f1:.3f}")
    print(f"Log-loss : {ll:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, preds, digits=3))

    out = test_df[["Date", "LeagueName", "HomeTeam", "AwayTeam", "FTR"]].copy()
    out["pred"] = preds
    out_path = Path("analysis_outputs") / "predictions_binary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Prediksi test disimpan ke: {out_path} ({len(out)} baris)")

    # simpan artefak
    home_stats, away_stats = build_team_snapshots(df_feat)
    artifact = {
        "model": pipe,
        "feature_cols": feature_cols,
        "home_stats": home_stats,
        "away_stats": away_stats,
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"Model + snapshot disimpan ke: {ARTIFACT_PATH}")


if __name__ == "__main__":
    train()

