# %% [markdown]
# # Baseline och första modeller: positivt inrikes flyttningsnetto nästa år
#
# **Mål:** Blir kommunens inrikes flyttningsnetto positivt nästa år (1) eller inte (0)?
# **Indata:** Bara sådant som är känt vid årets slut.
# **Uppdelning i tid:** Träning på målår 1999–2019, test på målår 2020–2024.
#
# Förutsätter att `python scripts/prepare_scb.py` har körts.

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

df = pd.read_csv("../data/processed/scb_kommun_ar.csv", dtype={"kod": str})
print(df.shape)
print("År:", df["ar"].min(), "-", df["ar"].max(), "| Kommuner:", df["kod"].nunique())

# %% [markdown]
# ## Variabler
# Absoluta tal beror på kommunens storlek, så allt delas med den inrikes
# omsättningen (inflyttningar + utflyttningar).

# %%
age_cols = [c for c in df.columns if c.startswith("inrikes_netto_")]

df["omsattning"] = df["inrikes_inflyttningar"] + df["inrikes_utflyttningar"]
df["netto_andel"] = df["inrikes_flyttningsnetto"] / df["omsattning"]
df["invandring_andel"] = df["invandringsnetto"] / df["omsattning"]
df["log_omsattning"] = np.log1p(df["omsattning"])   # storleksmått
df["positiv_i_ar"] = (df["inrikes_flyttningsnetto"] > 0).astype(int)
for c in age_cols:
    df[c.replace("inrikes_netto_", "andel_")] = df[c] / df["omsattning"]

g = df.groupby("kod")
df["netto_andel_forra"] = g["netto_andel"].shift(1)
df["forandring"] = df["netto_andel"] - df["netto_andel_forra"]
df["netto_andel_3ar"] = g["netto_andel"].transform(lambda s: s.rolling(3).mean())

# %% [markdown]
# ## Målvariabel: tecknet NÄSTA år
# Förskjuts ett år bakåt. Rader där nästa år saknas får inget mål.

# %%
df["mal"] = (g["inrikes_flyttningsnetto"].shift(-1) > 0).astype(float)
df.loc[g["ar"].shift(-1) != df["ar"] + 1, "mal"] = np.nan
df["mal_ar"] = df["ar"] + 1

FEATURES = (["netto_andel", "netto_andel_forra", "forandring", "netto_andel_3ar",
             "invandring_andel", "log_omsattning", "positiv_i_ar"]
            + [c.replace("inrikes_netto_", "andel_") for c in age_cols])

data = df.dropna(subset=FEATURES + ["mal"]).copy()
data["mal"] = data["mal"].astype(int)
print("Rader med fullständiga variabler och mål:", len(data))

# %% [markdown]
# ## Uppdelning i tid

# %%
train = data[data["mal_ar"] <= 2019]
test = data[data["mal_ar"] >= 2020]
X_train, y_train = train[FEATURES], train["mal"]
X_test, y_test = test[FEATURES], test["mal"]

print(f"Träning: {len(train)} rader, målår {train['mal_ar'].min()}-{train['mal_ar'].max()}, "
      f"andel positiva {y_train.mean():.1%}")
print(f"Test:    {len(test)} rader, målår {test['mal_ar'].min()}-{test['mal_ar'].max()}, "
      f"andel positiva {y_test.mean():.1%}")

# %% [markdown]
# ## Baselines och modeller

# %%
def evaluate(name, pred, prob=None):
    return {
        "modell": name,
        "accuracy": accuracy_score(y_test, pred),
        "balanced_acc": balanced_accuracy_score(y_test, pred),
        "f1_positiv": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, prob) if prob is not None else np.nan,
    }

results = []

# Baseline 1: vanligaste klassen i träningsdatan
majority = int(y_train.mode()[0])
results.append(evaluate(f"Baseline: alltid {majority}", np.full(len(y_test), majority)))

# Baseline 2: samma tecken som i år
results.append(evaluate("Baseline: samma tecken som i år", X_test["positiv_i_ar"]))

models = {
    "Logistisk regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Beslutsträd (djup 4)": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                            random_state=42, n_jobs=-1),
}
predictions = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    predictions[name] = pred
    results.append(evaluate(name, pred, prob))

results = pd.DataFrame(results).set_index("modell").round(3)
results

# %% [markdown]
# ## Per testår
# Pandemiåren kan skilja sig från resten. Hur håller modellerna varje år?

# %%
per_year = pd.DataFrame({"Samma tecken som i år": X_test["positiv_i_ar"].values == y_test.values,
                         **{n: p == y_test.values for n, p in predictions.items()}},
                        index=test["mal_ar"]).groupby(level=0).mean()
print(per_year.round(3))

per_year.plot(marker="o", figsize=(10, 4), title="Andel rätt per testår")
plt.ylabel("Accuracy")
plt.xticks(per_year.index)
plt.show()

# %% [markdown]
# ## Vad väger tyngst i logistisk regression?

# %%
logreg = models["Logistisk regression"]
coef = pd.Series(logreg[-1].coef_[0], index=FEATURES).sort_values()
coef.plot(kind="barh", figsize=(8, 6), title="Koefficienter (standardiserade variabler)")
plt.axvline(0, color="gray", linewidth=0.8)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Var gör modellen fel?
# Kommuner nära noll är svåra: -3 och +2 är i praktiken samma sak.

# %%
test_res = test[["kod", "namn", "mal_ar", "inrikes_flyttningsnetto", "mal"]].copy()
test_res["pred"] = predictions["Logistisk regression"]
test_res["fel"] = test_res["pred"] != test_res["mal"]

# Storleken på nästa års netto, för att se om felen ligger nära noll
next_netto = df.set_index(["kod", "ar"])["inrikes_flyttningsnetto"]
test_res["netto_nasta_ar"] = [next_netto.get((k, a)) for k, a in zip(test_res["kod"], test_res["mal_ar"])]

print("Median |netto nästa år| vid fel:  ", test_res.loc[test_res["fel"], "netto_nasta_ar"].abs().median())
print("Median |netto nästa år| vid rätt: ", test_res.loc[~test_res["fel"], "netto_nasta_ar"].abs().median())
print("\nKommuner med flest fel under testperioden:")
print(test_res.groupby("namn")["fel"].sum().sort_values(ascending=False).head(10))
# %%
