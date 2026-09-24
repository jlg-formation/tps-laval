"""Génère un historique synthétique (mais réaliste) des tickets quotidiens d'une DSI.

Le fichier produit contient volontairement des défauts (jours manquants, doublons,
valeurs vides, erreurs de saisie) qui seront détectés et corrigés par analyse.py.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "data" / "tickets_raw.csv"

START, END = date(2023, 1, 1), date(2025, 12, 31)
# Mise en service d'un nouvel outil de ticketing, qui capte davantage de demandes.
TOOL_CHANGE = date(2024, 11, 18)
# Panne de l'export : plusieurs jours consécutifs perdus.
EXPORT_OUTAGE = (date(2024, 3, 11), date(2024, 3, 20))

# Facteur multiplicatif par jour de la semaine (lundi -> dimanche).
WEEKDAY_FACTOR = np.array([1.35, 1.10, 1.00, 1.00, 0.90, 0.15, 0.08])

# Vacances scolaires zone B (dates approximatives).
SCHOOL_HOLIDAYS = [
    ("2022-12-17", "2023-01-02"),
    ("2023-02-11", "2023-02-26"),
    ("2023-04-15", "2023-05-01"),
    ("2023-07-08", "2023-09-03"),
    ("2023-10-21", "2023-11-05"),
    ("2023-12-23", "2024-01-07"),
    ("2024-02-24", "2024-03-10"),
    ("2024-04-20", "2024-05-05"),
    ("2024-07-06", "2024-09-01"),
    ("2024-10-19", "2024-11-03"),
    ("2024-12-21", "2025-01-05"),
    ("2025-02-08", "2025-02-23"),
    ("2025-04-05", "2025-04-21"),
    ("2025-07-05", "2025-08-31"),
    ("2025-10-18", "2025-11-02"),
    ("2025-12-20", "2026-01-04"),
]


def easter(year: int) -> date:
    """Dimanche de Pâques (algorithme de Meeus/Jones/Butcher)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def public_holidays(year: int) -> set[date]:
    fixed = [(1, 1), (5, 1), (5, 8), (7, 14), (8, 15), (11, 1), (11, 11), (12, 25)]
    e = easter(year)
    # Lundi de Pâques, Ascension, lundi de Pentecôte.
    movable = [e + timedelta(days=n) for n in (1, 39, 50)]
    return {date(year, mo, d) for mo, d in fixed} | set(movable)


rng = np.random.default_rng(42)
days = pd.date_range(START, END, freq="D")
n = len(days)

holidays = set().union(*(public_holidays(y) for y in range(START.year, END.year + 1)))
is_holiday = np.array([d.date() in holidays for d in days])
is_vacation = np.zeros(n, dtype=bool)
for first, last in SCHOOL_HOLIDAYS:
    is_vacation |= (days >= first) & (days <= last)
is_summer = is_vacation & days.month.isin([7, 8])

# Activité de base avec une croissance lente (~8 % par an).
level = 42 * (1 + 0.08 * np.arange(n) / 365)
lam = level * WEEKDAY_FACTOR[days.weekday]
lam[is_holiday] = level[is_holiday] * 0.1
lam *= np.where(is_summer, 0.7, np.where(is_vacation, 0.85, 1.0))
lam *= np.where(days >= pd.Timestamp(TOOL_CHANGE), 1.2, 1.0)

# Les demandes non traitées un jour férié se reportent sur le lendemain ouvré.
working = (WEEKDAY_FACTOR[days.weekday] > 0.5) & ~is_holiday
after_holiday = np.roll(is_holiday, 1)
after_holiday[0] = False
lam[after_holiday & working] *= 1.25

# Incidents majeurs : pic le jour même, contrecoup le lendemain.
incidents = np.sort(rng.choice(np.flatnonzero(working[:-1]), size=18, replace=False))
for i in incidents:
    boost = rng.uniform(1.8, 3.0)
    lam[i] *= boost
    lam[i + 1] *= 1 + (boost - 1) * 0.4

tickets = rng.poisson(lam)

# --- Défauts volontaires ---------------------------------------------------------
df = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "tickets": tickets.astype(str)})
outage = (days >= pd.Timestamp(EXPORT_OUTAGE[0])) & (days <= pd.Timestamp(EXPORT_OUTAGE[1]))
protected = outage.copy()
protected[incidents] = True
protected[incidents + 1] = True
pool = rng.permutation(np.flatnonzero(~protected))
missing, typos, empties, exact_dups, conflict_dups = np.split(pool[:35], [12, 17, 22, 30])

df.loc[typos, "tickets"] = ["-5", "9999", "-1", "4200", "999"]
df.loc[empties, "tickets"] = ""
dups = df.loc[exact_dups]
conflicts = df.loc[conflict_dups].copy()
conflicts["tickets"] = [str(int(v) + int(rng.integers(3, 15))) for v in conflicts["tickets"]]

drop = np.zeros(n, dtype=bool)
drop[missing] = True
drop |= outage
raw = pd.concat([df[~drop], dups, conflicts]).sort_values("date", kind="stable")

OUT.parent.mkdir(parents=True, exist_ok=True)
raw.to_csv(OUT, index=False)

print(f"{len(raw)} lignes écrites dans {OUT.relative_to(HERE.parent)}")
print(f"Période : {START} -> {END} ({n} jours)")
print("\nVérité terrain (pour l'enseignant) :")
print(f"  changement d'outil : {TOOL_CHANGE}")
print(f"  panne de l'export  : {EXPORT_OUTAGE[0]} -> {EXPORT_OUTAGE[1]}")
print(f"  incidents majeurs  : {', '.join(days[incidents].strftime('%Y-%m-%d'))}")
