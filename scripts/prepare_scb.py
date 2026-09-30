"""Bearbetar SCB:s flyttningsdata till en rad per kommun och år.

Förutsätter att rådata finns i data/raw/scb/ (skapas av EDA-notebooken).
Resultatet sparas i data/processed/scb_kommun_ar.csv.

Kör från projektroten:
    python scripts/prepare_scb.py
"""
import json
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "scb"
OUT = ROOT / "data" / "processed" / "scb_kommun_ar.csv"
API = "https://api.scb.se/OV0104/v1/doris/sv/ssd/START/BE/BE0101/BE0101J/Flyttningar97"

AGE_BINS = [-1, 17, 24, 34, 49, 64, 200]
AGE_LABELS = ["0_17", "18_24", "25_34", "35_49", "50_64", "65plus"]


def kommun_names() -> dict:
    """Hämtar kommunnamn från SCB:s metadata (ett enda snabbt anrop)."""
    try:
        with urllib.request.urlopen(API, timeout=30) as r:
            meta = json.loads(r.read().decode("utf-8-sig"))
        region = next(v for v in meta["variables"] if v["code"] == "Region")
        return dict(zip(region["values"], region["valueTexts"]))
    except Exception as e:  # namn är trevligt men inte nödvändigt
        print("Kunde inte hämta kommunnamn:", e)
        return {}


def main() -> None:
    # --- Totaler: alla åldrar och båda könen ---
    totals = pd.read_csv(RAW / "flyttningar_totaler.csv", dtype={"Region": str})
    totals = (totals.rename(columns={"Region": "kod", "Tid": "ar"})
                    .drop(columns="Alder", errors="ignore"))

    # År innan kommunen bildades (Knivsta, Nykvarn) har bara nollor
    measures = [c for c in totals.columns if c not in ("kod", "ar")]
    empty = totals[measures].fillna(0).eq(0).all(axis=1)
    print("Tar bort rader utan data:", empty.sum())
    print(totals.loc[empty].groupby("kod")["ar"].agg(["min", "max"]))
    totals = totals[~empty]

    # --- Åldrar: summera till sex åldersgrupper ---
    age = pd.read_csv(RAW / "flyttningar_alder.csv", dtype={"Region": str, "Alder": str})
    age["alder_num"] = age["Alder"].str.replace("+", "", regex=False).astype(int)
    age["grupp"] = pd.cut(age["alder_num"], AGE_BINS, labels=AGE_LABELS)
    age["netto"] = age["inrikes_inflyttningar"] - age["inrikes_utflyttningar"]

    age_wide = (age.groupby(["Region", "Tid", "grupp"], observed=True)["netto"].sum()
                   .unstack("grupp")
                   .add_prefix("inrikes_netto_")
                   .reset_index()
                   .rename(columns={"Region": "kod", "Tid": "ar"}))
    age_wide.columns.name = None

    df = totals.merge(age_wide, on=["kod", "ar"], how="left")

    # Kontroll: åldersgrupperna ska summera till totalen
    age_cols = [f"inrikes_netto_{g}" for g in AGE_LABELS]
    diff = (df[age_cols].sum(axis=1) - df["inrikes_flyttningsnetto"]).abs()
    print("Rader där åldersgrupperna inte summerar till totalen:", (diff > 0).sum())

    df.insert(1, "namn", df["kod"].map(kommun_names()))
    df = df.sort_values(["kod", "ar"]).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Sparade {df.shape[0]} rader och {df.shape[1]} kolumner till {OUT}")


if __name__ == "__main__":
    main()