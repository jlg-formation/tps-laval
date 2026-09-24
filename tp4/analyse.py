"""Analyse critique du fichier brut de tickets, puis nettoyage vers tickets.csv."""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "data" / "tickets_raw.csv"
CLEAN = HERE / "data" / "tickets.csv"
DAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

pd.options.display.width = 200


def fmt(day: pd.Timestamp) -> str:
    return day.strftime("%Y-%m-%d")


def runs(days: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Regroupe des dates triées en plages de jours consécutifs."""
    result: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for day in days:
        if result and day - result[-1][1] == pd.Timedelta(days=1):
            result[-1] = (result[-1][0], day)
        else:
            result.append((day, day))
    return result


def same_weekday_median(series: pd.Series, day: pd.Timestamp) -> float:
    """Médiane des mêmes jours de semaine voisins (J-7, J+7, puis J-14, J+14...)."""
    values: list[float] = []
    for weeks in range(1, 5):
        for sign in (-1, 1):
            value = series.get(day + pd.Timedelta(weeks=sign * weeks))
            if value is not None and not np.isnan(value):
                values.append(value)
        if len(values) >= 2:
            break
    return float(np.median(values))


raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
dates = pd.to_datetime(raw["date"], format="%Y-%m-%d")
values = pd.to_numeric(raw["tickets"], errors="coerce")
calendar = pd.date_range(dates.min(), dates.max(), freq="D")

print(f"== Fichier brut : {len(raw)} lignes, du {fmt(dates.min())} au {fmt(dates.max())}")
print(f"   Calendrier complet : {len(calendar)} jours")

# --- Valeurs vides -----------------------------------------------------------------
empty = values.isna()
print(f"\n== Valeurs vides : {empty.sum()}")
for day in dates[empty]:
    print(f"   {fmt(day)}")

# --- Doublons ----------------------------------------------------------------------
duplicated = dates.duplicated(keep=False)
print(f"\n== Doublons : {dates.duplicated().sum()} lignes en trop")
for day, group in raw[duplicated].groupby("date"):
    kind = "identiques" if group["tickets"].nunique() == 1 else "CONTRADICTOIRES"
    print(f"   {day} : {', '.join(group['tickets'])} ({kind})")

# --- Jours absents -----------------------------------------------------------------
absent = calendar.difference(dates)
print(f"\n== Jours absents du fichier : {len(absent)}")
for first, last in runs(absent):
    length = (last - first).days + 1
    print(f"   {fmt(first)}" + (f" -> {fmt(last)} ({length} jours)" if length > 1 else ""))

# --- Valeurs aberrantes ------------------------------------------------------------
# Au-delà de 10 fois la médiane (ou négatif), il s'agit d'une erreur de saisie, pas d'un incident.
max_plausible = 10 * values.median()
invalid = (values < 0) | (values > max_plausible)
print(f"\n== Valeurs impossibles (< 0 ou > {max_plausible:.0f}) : {invalid.sum()}")
for day, value in zip(dates[invalid], values[invalid]):
    print(f"   {fmt(day)} : {value:.0f}")

# --- Nettoyage ---------------------------------------------------------------------
series = (
    pd.DataFrame({"date": dates, "tickets": values.mask(invalid)})
    .groupby("date")["tickets"]
    .first()  # en cas de doublon contradictoire, on garde la première saisie
    .reindex(calendar)
)
to_impute = series.index[series.isna()]
clean = series.copy()
for day in to_impute:
    clean[day] = round(same_weekday_median(series, day))
clean = clean.astype(int)

print(f"\n== Jours reconstitués (médiane des mêmes jours voisins) : {len(to_impute)}")

# --- Pics d'activité (conservés) ---------------------------------------------------
reference = clean.groupby(clean.index.weekday).transform(
    lambda s: s.rolling(9, center=True, min_periods=3).median()
)
spikes = clean[(clean > 1.6 * reference) & (clean > 30)]
print(f"\n== Pics d'activité probables (incidents ?) : {len(spikes)} — conservés")
for day, value in spikes.items():
    print(f"   {fmt(day)} ({DAYS[day.weekday()]}) : {value} (habituel ≈ {reference[day]:.0f})")

# --- Profil hebdomadaire -----------------------------------------------------------
print("\n== Moyenne par jour de la semaine")
for weekday, mean in clean.groupby(clean.index.weekday).mean().items():
    print(f"   {DAYS[weekday]:<9} {mean:6.1f}")

# --- Évolution mensuelle et rupture de niveau --------------------------------------
monthly = clean.resample("MS").mean()
table = monthly.groupby([monthly.index.year, monthly.index.month]).first().unstack().round(1)
table.index.name, table.columns.name = "année", "mois"
print("\n== Moyenne quotidienne par mois")
print(table.to_string())

yoy = monthly / monthly.shift(12)
jump = yoy.diff()
month = jump.idxmax()
print("\n== Croissance sur un an (même mois, année précédente)")
print("   " + "  ".join(f"{m:%Y-%m} {r - 1:+.0%}" for m, r in yoy.dropna().items()))
print(
    f"   Rupture probable autour de {month:%Y-%m} "
    f"({yoy[month - pd.DateOffset(months=1)] - 1:+.0%} -> {yoy[month] - 1:+.0%}) :"
    " changement d'outil ? Non corrigé, à discuter."
)

clean.rename_axis("date").rename("tickets").to_frame().to_csv(CLEAN, date_format="%Y-%m-%d")
print(f"\n{len(clean)} jours écrits dans {CLEAN.relative_to(HERE.parent)}")
