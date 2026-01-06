"""
Enhanced visualizations from analysis outputs (and main dataset).

Inputs:
  - analysis_outputs/total_matches.csv
  - analysis_outputs/ftr_dist.csv
  - analysis_outputs/avg_goals.csv
  - analysis_outputs/top_winrate.csv
  - analysis_outputs/home_adv.csv
  - ALL_leagues_preprocessed.csv (for extra monthly/weekday charts)

Outputs:
  - analysis_outputs/plots/*.png

Usage:
  python visualize_analysis.py

Requires:
  pip install pandas seaborn matplotlib
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path("analysis_outputs")
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

DATA_MAIN = Path("ALL_leagues_preprocessed.csv")


# Color palette menggunakan warna yang ditentukan user
LEAGUE_PALETTE = {
    "EPL": "#0077b6",      # Biru gelap untuk EPL
    "LaLiga": "#48cae4",   # Biru muda/cyan untuk LaLiga
    "SerieA": "#a4133c",   # Merah gelap untuk SerieA
    "Bundes": "#ff4d6d",   # Merah muda/pink untuk Bundes
}

# Result colors (Home/Draw/Away) - menggunakan kombinasi warna yang sama
RESULT_COLORS = {
    "H": "#0077b6",  # Biru gelap untuk Home Win
    "D": "#48cae4",  # Biru muda untuk Draw
    "A": "#a4133c",  # Merah gelap untuk Away Win
}


def setup_style():
    """Setup modern, minimal styling for all plots"""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


def savefig(name: str, bbox_inches='tight', pad_inches=0.3):
    path = PLOT_DIR / name
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(path, dpi=300, bbox_inches=bbox_inches, pad_inches=pad_inches, facecolor='white')
    print(f"Saved plot: {path}")
    plt.close()


def plot_ftr_distribution():
    df = pd.read_csv(BASE_DIR / "ftr_dist.csv")
    pivot = df.pivot(index="LeagueName", columns="FTR", values="pct").fillna(0)[["H", "D", "A"]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create grouped bar chart (not stacked)
    x = range(len(pivot.index))
    width = 0.25
    colors = RESULT_COLORS
    labels_map = {"H": "Home Win", "D": "Draw", "A": "Away Win"}
    
    for i, res in enumerate(["H", "D", "A"]):
        offset = (i - 1) * width
        bars = ax.bar([xi + offset for xi in x], pivot[res], 
                      width=width, label=labels_map[res], 
                      color=colors[res], edgecolor='white', linewidth=1.5)
        
        # Add percentage labels on bars
        for j, (league, val) in enumerate(zip(pivot.index, pivot[res])):
            if val > 0:
                ax.text(j + offset, val + 1, f"{val:.1f}%", 
                       ha="center", va="bottom", 
                       fontsize=9, fontweight='600', color='#1F2937')
    
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, fontsize=11)
    ax.set_ylabel("Share of Results (%)", fontweight='500', fontsize=12)
    ax.set_xlabel("", fontweight='500', fontsize=12)
    ax.set_title("Result Distribution by League", fontweight='600', fontsize=15, pad=20)
    ax.set_ylim(0, max(pivot.max()) * 1.2)
    ax.legend(title="Result Type", frameon=True, fancybox=True, shadow=True, 
              title_fontsize=11, fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    
    savefig("ftr_distribution.png")


def plot_avg_goals_trend():
    df = pd.read_csv(BASE_DIR / "avg_goals.csv")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot all leagues in one chart
    for league in df["LeagueName"].unique():
        league_data = df[df["LeagueName"] == league].sort_values("season_year")
        color = LEAGUE_PALETTE.get(league, "#6B7280")
        ax.plot(league_data["season_year"], league_data["avg_total_goals"], 
               marker="o", markersize=5, linewidth=2.5, 
               color=color, label=league)
    
    # Format x-axis labels to 05/06 format
    unique_years = sorted(df["season_year"].unique())
    ax.set_xticks(unique_years)
    season_labels = []
    for y in unique_years:
        year_str = str(int(y))
        next_year_str = str(int(y) + 1)
        season_labels.append(f"{year_str[-2:]}/{next_year_str[-2:]}")
    ax.set_xticklabels(season_labels, rotation=45, ha='right', fontsize=9)
    
    ax.set_xlabel("", fontweight='500', fontsize=12)
    ax.set_ylabel("Average Total Goals", fontweight='500', fontsize=12)
    ax.set_title("Average Total Goals per Season", fontweight='600', fontsize=15, pad=20)
    ax.legend(title="League", frameon=True, fancybox=True, shadow=True, 
             title_fontsize=11, fontsize=10, loc='best')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    savefig("avg_goals_trend.png")


def plot_home_advantage():
    df = pd.read_csv(BASE_DIR / "home_adv.csv")
    df_sorted = df.sort_values("home_advantage_points", ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create bars with league colors (using numeric y positions, sama seperti top_winrate)
    y_positions = range(len(df_sorted))
    colors = [LEAGUE_PALETTE.get(league, "#6B7280") for league in df_sorted["LeagueName"]]
    
    bars = ax.barh(y_positions, df_sorted["home_advantage_points"], 
                   color=colors, edgecolor='white', linewidth=2, height=0.85)
    
    # Add league name and value inside bars at left edge (sama seperti top_winrate)
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        width = row["home_advantage_points"]
        league_name = row["LeagueName"]
        # Position text at left edge inside the bar (starting from 0.35 + small offset)
        text_x = 0.35 + 0.005  # Small offset from 0.35 to be inside the bar
        ax.text(text_x, i, f"{league_name} {width:.3f}", 
               va="center", ha="left", fontsize=11, fontweight='400', color='black')
    
    # Create custom legend (sama seperti top_winrate)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=LEAGUE_PALETTE[league], edgecolor='white', linewidth=1.5) 
                      for league in LEAGUE_PALETTE.keys()]
    ax.legend(legend_elements, LEAGUE_PALETTE.keys(), 
             title="League", frameon=True, fancybox=True, shadow=True,
             title_fontsize=11, fontsize=10, loc='lower right')
    
    # Remove y-axis labels and ticks (sama seperti top_winrate)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel("", fontweight='500', fontsize=12)
    ax.set_ylabel("", fontweight='500', fontsize=12)
    ax.set_title("Home Advantage by League", fontweight='600', fontsize=15, pad=20)
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    ax.set_xlim(0.35, 0.6)
    
    savefig("home_advantage.png")


def plot_top_winrate():
    df = pd.read_csv(BASE_DIR / "top_winrate.csv")
    top10 = df.sort_values("win_rate_pct", ascending=False).head(10)
    top10_sorted = top10.sort_values("win_rate_pct", ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bars with league colors (using numeric y positions)
    y_positions = range(len(top10_sorted))
    colors = [LEAGUE_PALETTE.get(league, "#6B7280") for league in top10_sorted["LeagueName"]]
    
    bars = ax.barh(y_positions, top10_sorted["win_rate_pct"], 
                   color=colors, edgecolor='white', linewidth=2, height=0.85)
    
    # Add team name and percentage inside bars at left edge (starting from 53%)
    for i, (idx, row) in enumerate(top10_sorted.iterrows()):
        width = row["win_rate_pct"]
        team_name = row["Team"]
        # Position text at left edge inside the bar (just after 53% mark)
        text_x = 53 + 0.3  # Small offset from 53% to be inside the bar
        ax.text(text_x, i, f"{team_name} {width:.1f}%", 
               va="center", ha="left", fontsize=11, fontweight='400', color='black')
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=LEAGUE_PALETTE[league], edgecolor='white', linewidth=1.5) 
                      for league in LEAGUE_PALETTE.keys()]
    ax.legend(legend_elements, LEAGUE_PALETTE.keys(), 
             title="League", frameon=True, fancybox=True, shadow=True,
             title_fontsize=11, fontsize=10, loc='lower right')
    
    # Remove y-axis labels and ticks
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_title("Top 10 Teams by Win Rate", fontweight='600', fontsize=15, pad=20)
    ax.grid(axis='x', alpha=0.2, linestyle='--')
    ax.set_xlim(53, 71)
    
    # Format x-axis ticks to remove decimal points
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
    
    savefig("top_winrate_top10.png")


def plot_avg_goals_home_away():
    df = pd.read_csv(BASE_DIR / "avg_goals.csv")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot all leagues in one chart
    for league in df["LeagueName"].unique():
        league_data = df[df["LeagueName"] == league].sort_values("season_year")
        color = LEAGUE_PALETTE.get(league, "#6B7280")
        
        # Home goals - solid line
        ax.plot(league_data["season_year"], league_data["avg_home_goals"], 
               marker="o", markersize=4, linewidth=2, 
               color=color, linestyle="-", label=f"{league} (Home)")
        
        # Away goals - dashed line
        ax.plot(league_data["season_year"], league_data["avg_away_goals"], 
               marker="s", markersize=4, linewidth=2, 
               color=color, linestyle="--", label=f"{league} (Away)")
    
    # Format x-axis labels to 05/06 format (sama seperti avg_goals_trend.png)
    unique_years = sorted(df["season_year"].unique())
    ax.set_xticks(unique_years)
    season_labels = []
    for y in unique_years:
        year_str = str(int(y))
        next_year_str = str(int(y) + 1)
        season_labels.append(f"{year_str[-2:]}/{next_year_str[-2:]}")
    ax.set_xticklabels(season_labels, rotation=45, ha='right', fontsize=9)
    
    ax.set_xlabel("", fontweight='500', fontsize=12)
    ax.set_ylabel("Average Goals", fontweight='500', fontsize=12)
    ax.set_title("Home vs Away Average Goals per Season", fontweight='600', fontsize=15, pad=20)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=9, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    savefig("avg_goals_home_away.png")




def main():
    setup_style()
    sns.set_style("whitegrid", {'axes.grid': True, 'grid.alpha': 0.3})
    
    # Plots based on precomputed aggregates
    plot_ftr_distribution()
    plot_avg_goals_trend()
    plot_home_advantage()
    plot_top_winrate()
    plot_avg_goals_home_away()

    print("All plots generated in analysis_outputs/plots/")


if __name__ == "__main__":
    main()

