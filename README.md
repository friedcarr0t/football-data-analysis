# Football Match Data ETL and Analytics (2005–2025)

## Project Overview
This project implements an end-to-end **ETL (Extract, Transform, Load)** and **data analytics workflow** on historical football match data from four major European leagues: **English Premier League (EPL), Bundesliga, La Liga, and Serie A**, covering seasons from **2005 to 2025**.

The main objective of this project is to demonstrate data analytics capabilities, including data collection, cleaning, transformation, exploratory data analysis (EDA), and insight generation from a large, multi-year dataset using Python-based analytical tools.

---

## Dataset
The dataset is sourced from **https://football-data.co.uk** and consists of historical match results from top European football leagues.

**Dataset summary:**
- Total matches: **29,479**
- Total columns: **25**
- Leagues included:
  - English Premier League (EPL): 7,750 matches
  - Bundesliga: 6,237 matches
  - La Liga: 7,752 matches
  - Serie A: 7,740 matches

Each record represents a single football match and includes information such as match result, home and away goals, and league identifiers.

---

## Tools & Technologies
- **Python**
- **Pandas** for data manipulation
- **DuckDB** for analytical SQL queries
- **Matplotlib** for data visualization

---

## ETL Workflow
The ETL pipeline is structured as follows:

1. **Extract**
   - Load raw CSV files for multiple leagues and seasons from local storage.

2. **Transform**
   - Standardize column names and schemas across leagues.
   - Handle missing values and inconsistent data formats.
   - Merge multi-season and multi-league data into a unified dataset.

3. **Load**
   - Store transformed data into an analytical environment using Pandas and DuckDB for further analysis.

A visual representation of the workflow is provided in `diagramAlur.png`.

---

## Analytical Outputs
The analysis focuses on league-level and season-level performance trends, including:

- Average **home vs away goals per season** for each league
- Average **total goals per season** by league
- Distribution of **full-time match results** (home win, draw, away win)
- **Home advantage win percentage** across leagues
- **Top 10 teams** with the highest overall win rate across all leagues

---

## Key Insights
Key findings derived from the analysis include:

- Home advantage exists across all leagues, but its impact varies by league and season.
- Bundesliga consistently shows the **highest and most stable average goals per match**, indicating a more open playing style.
- Across the entire dataset, **FC Barcelona** demonstrates the highest long-term win rate among all teams included.

These insights highlight how historical match data can be used to identify long-term performance patterns and league characteristics.

---

## How to Run the Project

### Requirements
Install the required Python libraries:
```bash
pip install pandas duckdb matplotlib
```

---

### Execution Steps
1. Place raw CSV files from football-data.co.uk into:
```bash
dataset/raw_csvs/
```
2. Run the preprocessing script:
```bash
python 1.\ preprocess.py
```
3. Run the analysis script:
```bash
python 2.\ analysis.py
```
4. Generate visualizations:
```bash
python 3.\ visualize.py
```
---

## Outputs
- Generated plots and figures are saved in the `analysis_outputs/` directory.
- Analytical summaries are printed during execution and can be extended for export if needed.
---
## Limitations
- Analysis is performed at match and team level only; no player-level data is included.
- Statistical hypothesis testing is not implemented and could be added for deeper analysis.

---
