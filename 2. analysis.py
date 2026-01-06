"""
Starter analysis for ALL_leagues_preprocessed.csv using DuckDB + pandas.

What it does:
- Load the merged dataset
- Register as DuckDB table
- Run key aggregates (total matches, FTR distribution, avg goals, top teams)
- Save aggregate outputs to CSV under ./analysis_outputs/

Requirements:
- pip install duckdb pandas

Usage:
    python analysis_starter.py
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd


DATA_PATH = Path("ALL_leagues_preprocessed.csv")
OUT_DIR = Path("analysis_outputs")


def ensure_outdir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def run_queries(df: pd.DataFrame):
    con = duckdb.connect()
    con.register("matches", df)

    # Total matches per league / season (cast Date to DATE)
    total_matches = con.execute(
        """
        SELECT
          LeagueName,
          EXTRACT(YEAR FROM CAST(Date AS DATE)) AS season_year,
          COUNT(*) AS matches
        FROM matches
        GROUP BY LeagueName, season_year
        ORDER BY LeagueName, season_year
        """
    ).df()

    # FTR distribution (H/D/A) per league
    ftr_dist = con.execute(
        """
        SELECT
          LeagueName,
          FTR,
          COUNT(*) AS cnt,
          ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY LeagueName), 2) AS pct
        FROM matches
        GROUP BY LeagueName, FTR
        ORDER BY LeagueName, cnt DESC
        """
    ).df()

    # Average goals (full time) per league / season
    avg_goals = con.execute(
        """
        SELECT
          LeagueName,
          EXTRACT(YEAR FROM CAST(Date AS DATE)) AS season_year,
          AVG(FTHG) AS avg_home_goals,
          AVG(FTAG) AS avg_away_goals,
          AVG(FTHG + FTAG) AS avg_total_goals
        FROM matches
        GROUP BY LeagueName, season_year
        ORDER BY LeagueName, season_year
        """
    ).df()

    # Top teams by win rate (overall, min 30 matches)
    top_winrate = con.execute(
        """
        WITH team_results AS (
          SELECT
            LeagueName,
            HomeTeam AS Team,
            CASE WHEN FTR = 'H' THEN 1 ELSE 0 END AS win,
            1 AS played
          FROM matches
          UNION ALL
          SELECT
            LeagueName,
            AwayTeam AS Team,
            CASE WHEN FTR = 'A' THEN 1 ELSE 0 END AS win,
            1 AS played
          FROM matches
        )
        SELECT
          LeagueName,
          Team,
          SUM(win) AS wins,
          SUM(played) AS played,
          ROUND(100.0 * SUM(win) / NULLIF(SUM(played),0), 2) AS win_rate_pct
        FROM team_results
        GROUP BY LeagueName, Team
        HAVING SUM(played) >= 30
        ORDER BY win_rate_pct DESC, wins DESC
        LIMIT 50
        """
    ).df()

    # Home advantage: average points home vs away per league
    home_adv = con.execute(
        """
        WITH points AS (
          SELECT
            LeagueName,
            CASE
              WHEN FTR = 'H' THEN 3
              WHEN FTR = 'D' THEN 1
              ELSE 0
            END AS home_pts,
            CASE
              WHEN FTR = 'A' THEN 3
              WHEN FTR = 'D' THEN 1
              ELSE 0
            END AS away_pts
          FROM matches
        )
        SELECT
          LeagueName,
          AVG(home_pts) AS avg_home_points_per_match,
          AVG(away_pts) AS avg_away_points_per_match,
          (AVG(home_pts) - AVG(away_pts)) AS home_advantage_points
        FROM points
        GROUP BY LeagueName
        ORDER BY home_advantage_points DESC
        """
    ).df()

    return {
        "total_matches": total_matches,
        "ftr_dist": ftr_dist,
        "avg_goals": avg_goals,
        "top_winrate": top_winrate,
        "home_adv": home_adv,
    }


def save_outputs(results: dict):
    for name, df in results.items():
        out_path = OUT_DIR / f"{name}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path} ({len(df)} rows)")


def main():
    ensure_outdir()
    df = load_data()
    results = run_queries(df)
    save_outputs(results)
    print("Done.")


if __name__ == "__main__":
    main()

