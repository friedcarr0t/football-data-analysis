"""
Merge dan preprocess multi-liga (EPL, Bundesliga, La Liga, Serie A) dari folder:
  - EPL/
  - Bundes/
  - LaLiga/
  - SerieA/

Langkah:
1) Baca seluruh CSV di tiap folder.
2) Normalisasi tanggal ke YYYY-MM-DD.
3) Filter rentang tahun 2005-01-01 s/d 2025-12-31.
4) Pilih hanya kolom penting untuk modeling.
5) Simpan ke ALL_leagues_preprocessed.csv

Jalankan dari root proyek:
    python merge_and_preprocess_leagues.py
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List

LEAGUE_DIRS = {
    "EPL": Path("EPL"),
    "Bundes": Path("Bundes"),
    "LaLiga": Path("LaLiga"),
    "SerieA": Path("SerieA"),
}

DATE_MIN = datetime(2005, 1, 1)
DATE_MAX = datetime(2025, 12, 31)

REQUIRED_COLS = [
    "Div",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG",
    "FTAG",
    "FTR",
    "HTHG",
    "HTAG",
    "HTR",
    "Attendance",
    "Referee",
    "HS",
    "AS",
    "HST",
    "AST",
    "HHW",
    "AHW",
    "HC",
    "AC",
    "HF",
    "AF",
    "HO",
    "AO",
    "HY",
    "AY",
    "HR",
    "AR",
]


def normalize_date(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    fmts = ["%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d", "%d-%m-%Y", "%d-%m-%y"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            if fmt in ("%d/%m/%y", "%d-%m-%y") and dt.year > 2050:
                dt = dt.replace(year=dt.year - 100)
            return dt.date()
        except Exception:
            continue
    return None


def read_csv_safe(path: Path) -> pd.DataFrame | None:
    encs = ["utf-8", "latin-1", "iso-8859-1", "cp1252"]
    for enc in encs:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", engine="python")
        except Exception:
            try:
                return pd.read_csv(path, encoding=enc)
            except Exception:
                continue
    return None


def load_league(name: str, folder: Path) -> List[pd.DataFrame]:
    frames = []
    for csv_path in sorted(folder.glob("*.csv")):
        df = read_csv_safe(csv_path)
        if df is None or df.empty:
            print(f"Skip {csv_path.name} (gagal/empty)")
            continue
        df["SourceFile"] = csv_path.name
        df["LeagueName"] = name
        if "Date" in df.columns:
            df["Date"] = df["Date"].apply(normalize_date)
        frames.append(df)
        print(f"OK  {name}: {csv_path.name} -> {len(df)} baris")
    return frames


def main():
    all_frames: List[pd.DataFrame] = []
    for name, folder in LEAGUE_DIRS.items():
        if not folder.exists():
            print(f"Folder tidak ditemukan: {folder}")
            continue
        all_frames.extend(load_league(name, folder))

    if not all_frames:
        print("Tidak ada data yang terbaca.")
        return

    merged = pd.concat(all_frames, ignore_index=True)

    # filter tahun
    if "Date" in merged.columns:
        before = len(merged)
        merged = merged[merged["Date"].notna()]
        merged = merged[(merged["Date"] >= DATE_MIN.date()) & (merged["Date"] <= DATE_MAX.date())]
        print(f"Filter tahun 2005-2025: hapus {before - len(merged)} baris")

    # pilih kolom
    cols_keep = [c for c in REQUIRED_COLS if c in merged.columns]
    extra_cols = ["SourceFile", "LeagueName"]
    cols_final = cols_keep + extra_cols
    merged = merged[cols_final]

    # drop baris kosong pada kolom penting
    important = [c for c in ["HomeTeam", "AwayTeam", "FTHG", "FTAG"] if c in merged.columns]
    if important:
        before = len(merged)
        merged = merged.dropna(subset=important, how="all")
        print(f"Hapus baris kosong penting: {before - len(merged)}")

    # drop baris all-NaN
    before = len(merged)
    merged = merged.dropna(how="all")
    print(f"Hapus baris all-NaN: {before - len(merged)}")

    # drop kolom kosong total
    before_cols = len(merged.columns)
    merged = merged.dropna(axis=1, how="all")
    print(f"Hapus kolom kosong total: {before_cols - len(merged.columns)}")

    # sort tanggal
    if "Date" in merged.columns:
        merged = merged.sort_values("Date", na_position="last").reset_index(drop=True)

    out_file = Path("ALL_leagues_preprocessed.csv")
    merged.to_csv(out_file, index=False)

    print("\nSelesai.")
    print(f"  Output: {out_file}")
    print(f"  Baris : {len(merged):,}")
    print(f"  Kolom : {len(merged.columns)}")
    if "LeagueName" in merged.columns:
        print("  Baris per liga:")
        print(merged["LeagueName"].value_counts().sort_index())


if __name__ == "__main__":
    main()


