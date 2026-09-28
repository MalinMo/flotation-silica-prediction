# flotation-silica-prediction
Gruppinlämningsuppgift

# Flotation Silica Prediction

Prognos av kiselhalt (% Silica Concentrate) i järnmalmskoncentrat från en flotationsprocess.

> **Status:** Preliminärt projektförslag – under utveckling i kursen *Projekt i Data Science*.

## Problem

Kiselhalten i koncentratet är ett centralt kvalitetsmått, men den mäts i labb och svaret kommer med fördröjning. Om halten kan förutsägas i förväg kan processingenjörer och operatörer agera tidigare, till exempel för att minska orenheter och mängden järn som går förlorad till avfall (tailings).

**Preliminär frågeställning:** Hur många timmar framåt går det att förutsäga kiselhalten bättre än en enkel baseline?

## Tänkt användare

Processingenjörer och operatörer i en flotationsanläggning.

## Data

| | |
|---|---|
| **Dataset** | Quality Prediction in a Mining Process |
| **Källa** | [Kaggle](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process) |
| **Ansvarig** | Eduardo Magalhães Oliveira (Kaggle-användare `edumagalhaes`) |
| **Licens** | CC0: Public Domain |
| **Hämtad** | 2026-09-28 via `kagglehub` |
| **Tidsperiod** | Mars–september 2017 |
| **Fil** | `MiningProcess_Flotation_Plant_Database.csv` |

**Innehåll:** Processdata från en verklig flotationsanläggning: kvalitetsmått på malmen före flotation, styrvariabler (bl.a. flöden och doseringar), nivåer och luftflöden i flotationskolonnerna samt labbmätningar av järn- och kiselhalt i koncentratet.

**Kända begränsningar (preliminärt, ska verifieras):**
- Kolumnerna har olika samplingsfrekvens (vissa var 20:e sekund, andra en gång per timme). Tidsstämpeln har timupplösning, vilket ger många rader med samma tidsstämpel och upprepade timvärden.
- Möjliga luckor i tidsserien.
- `% Iron Concentrate` mäts samtidigt som målvariabeln och riskerar att orsaka dataläckage om den används som indata.
- Filen använder decimalkomma.

Datan versionshanteras inte i repot, se *Kom igång* nedan.

## Kom igång

```bash
git clone https://github.com/MalinMo/flotation-silica-prediction.git
cd flotation-silica-prediction

python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash)
# source .venv/bin/activate        # macOS/Linux

python -m pip install -e ".[dev]"
python scripts/download_data.py    # laddar ner data till data/raw/
```

## Projektstruktur - uppdateras löpande

```
flotation-silica-prediction/
├── data/                  # ignoreras av git
│   └── raw/
├── notebooks/             # utforskning, en notebook per person/uppgift
├── scripts/               # körbara steg i arbetsflödet
│   └── download_data.py
├── src/
│   └── flotation_silica_prediction/   # återanvändbar kod
├── pyproject.toml
└── README.md
```

## Arbetssätt

- Delad kod läggs i `src/` och körbara steg i `scripts/`.
- Notebooks är personliga och används för utforskning.
- Nya beroenden läggs till i `pyproject.toml`.