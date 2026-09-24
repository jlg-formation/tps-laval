# TP4 — Prévision des tickets d'une DSI avec un RNN

Sujet : [brief.md](brief.md).

## Chaîne complète

```bash
mise run tp4-data    # generate_data.py -> data/tickets_raw.csv, analyse.py -> data/tickets.csv
mise run tp4-train   # train.py -> ticket-rnn.pth, public/ticket-rnn.onnx, public/ticket-rnn.json
mise run tp4         # application Web : http://localhost:5173
```

| Script | Rôle |
| --- | --- |
| `generate_data.py` | Historique synthétique 2023-2025 : jours de semaine, week-ends, fériés, vacances, tendance, incidents, changement d'outil. Défauts volontaires : trous, doublons, valeurs vides, erreurs de saisie. |
| `analyse.py` | Rapport de qualité, nettoyage (dédoublonnage, suppression des valeurs impossibles, reconstitution des jours manquants), profil hebdomadaire, rupture de niveau. |
| `train.py` | Séquences, découpage chronologique 80/20, entraînement, comparaison aux méthodes naïves, export ONNX (normalisation incluse) et vérification de parité. |

## Expériences

```bash
mise run tp4-train --window 14
mise run tp4-train --cell gru --window 30
mise run tp4-train --cell lstm --hidden 32 --epochs 300
```

Chaque entraînement remplace le modèle utilisé par l'application Web (le formulaire s'adapte à la taille de fenêtre).

## Questions de discussion

- Le RNN fait-il mieux que « même jour la semaine précédente » (J-7) ? Pourquoi ?
- Combien d'historique faut-il ? Une fenêtre de 7 jours suffit-elle ?
- Comment prendre en compte les jours fériés, les vacances, les mises en production ?
- Faut-il corriger la rupture due au changement d'outil, ou n'entraîner que sur les données récentes ?
- Peut-on prévoir un incident sans signe précurseur ?
- GRU ou LSTM apportent-ils quelque chose sur ce problème ?
- Quelle erreur (MAE, RMSE) serait acceptable pour que la prévision soit utile à la DSI ?
- Que faudrait-il changer pour passer à `input_size=5` (série multivariée) ?
