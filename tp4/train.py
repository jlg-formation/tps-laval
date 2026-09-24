"""Entraîne un RNN à prévoir le nombre de tickets du lendemain, puis l'exporte en ONNX."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).parent
DATA = HERE / "data" / "tickets.csv"
PUBLIC = HERE / "public"
DAYS = ["lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim."]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--window", type=int, default=7, help="nombre de jours en entrée (>= 7)")
parser.add_argument("--cell", choices=["rnn", "gru", "lstm"], default="rnn")
parser.add_argument("--hidden", type=int, default=16)
parser.add_argument("--epochs", type=int, default=150)
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
if args.window < 7:
    parser.error("--window doit valoir au moins 7 (la référence J-7 en a besoin)")
torch.manual_seed(args.seed)


class TicketRNN(nn.Module):
    """Séquence normalisée (batch, window, 1) -> tickets du lendemain normalisés (batch, 1)."""

    def __init__(self, cell: str, hidden: int):
        super().__init__()
        rnn_class = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_class(input_size=1, hidden_size=hidden, batch_first=True)
        self.linear = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.linear(out[:, -1, :])


class DeployedModel(nn.Module):
    """Intègre la normalisation au modèle exporté : l'application envoie des tickets bruts."""

    def __init__(self, model: nn.Module, mean: float, std: float):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x):
        return self.model((x - self.mean) / self.std) * self.std + self.mean


# --- Séquences -----------------------------------------------------------------------
df = pd.read_csv(DATA, parse_dates=["date"])
dates = df["date"]
series = df["tickets"].to_numpy(dtype=np.float32)

# Découpage chronologique : le test est la fin de l'historique, jamais mélangée au passé.
split = int(len(series) * 0.8)
mean, std = float(series[:split].mean()), float(series[:split].std())


def windows(targets: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    X = np.stack([series[t - args.window : t] for t in targets])[:, :, None]
    y = series[targets][:, None]
    return torch.from_numpy(X), torch.from_numpy(y)


train_t = np.arange(args.window, split)
test_t = np.arange(split, len(series))
X_train, y_train = windows(train_t)
X_test, y_test = windows(test_t)
print(f"Entraînement : {len(train_t)} séquences ({dates[0]:%Y-%m-%d} -> {dates[split - 1]:%Y-%m-%d})")
print(f"Test         : {len(test_t)} séquences ({dates[split]:%Y-%m-%d} -> {dates.iloc[-1]:%Y-%m-%d})")

# --- Entraînement --------------------------------------------------------------------
model = TicketRNN(args.cell, args.hidden)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# Mélanger les séquences d'entraînement entre elles ne fait pas fuiter le futur.
loader = DataLoader(
    TensorDataset((X_train - mean) / std, (y_train - mean) / std), batch_size=32, shuffle=True
)

print(f"\nModèle {args.cell.upper()} : fenêtre {args.window} jours, {args.hidden} neurones cachés")
for epoch in range(1, args.epochs + 1):
    model.train()
    total = 0.0
    for X, y in loader:
        prediction = model(X)
        loss = criterion(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(X)
    if epoch == 1 or epoch % 25 == 0:
        print(f"  époque {epoch:4d}  perte (MSE normalisée) : {total / len(X_train):.4f}")

# --- Évaluation ----------------------------------------------------------------------
deployed = DeployedModel(model, mean, std).eval()
with torch.no_grad():
    predicted = deployed(X_test).squeeze(1).numpy()
actual = y_test.squeeze(1).numpy()

forecasts = {
    f"{args.cell.upper()} (fenêtre {args.window})": predicted,
    "Veille (J-1)": series[test_t - 1],
    "Semaine précédente (J-7)": series[test_t - 7],
    "Moyenne des 7 derniers jours": np.array([series[t - 7 : t].mean() for t in test_t]),
}
metrics = [
    {
        "name": name,
        "mae": float(np.abs(forecast - actual).mean()),
        "rmse": float(np.sqrt(((forecast - actual) ** 2).mean())),
    }
    for name, forecast in forecasts.items()
]

print(f"\n{'Méthode':<32}{'MAE':>8}{'RMSE':>8}")
for m in metrics:
    print(f"{m['name']:<32}{m['mae']:8.2f}{m['rmse']:8.2f}")

print(f"\n{'Jour':<16}{'Réel':>6}{'Prédit':>8}{'Écart':>7}")
for t, real, pred in list(zip(test_t, actual, predicted))[:14]:
    day = dates[t]
    print(f"{DAYS[day.weekday()]} {day:%Y-%m-%d}  {real:6.0f}{pred:8.1f}{pred - real:+7.1f}")

# --- Sauvegardes ---------------------------------------------------------------------
pth_path = HERE / "ticket-rnn.pth"
torch.save(
    {
        "state_dict": model.state_dict(),
        "cell": args.cell,
        "hidden": args.hidden,
        "window": args.window,
        "mean": mean,
        "std": std,
    },
    pth_path,
)

PUBLIC.mkdir(parents=True, exist_ok=True)
onnx_path = PUBLIC / "ticket-rnn.onnx"
torch.onnx.export(
    deployed,
    (torch.zeros(1, args.window, 1),),
    str(onnx_path),
    input_names=["input"],
    output_names=["output"],
    external_data=False,
)

# Vérifie que le modèle ONNX donne les mêmes prévisions que PyTorch.
session = ort.InferenceSession(str(onnx_path))
onnx_predicted = np.array([session.run(None, {"input": x[None]})[0][0, 0] for x in X_test.numpy()])
print(f"\nÉcart max PyTorch / ONNX : {np.abs(onnx_predicted - predicted).max():.2e}")

last_values = series[-args.window :]
next_prediction = float(session.run(None, {"input": last_values[None, :, None]})[0][0, 0])
info = {
    "cell": args.cell,
    "window": args.window,
    "hidden": args.hidden,
    "metrics": metrics,
    "test": {
        "dates": [f"{dates[t]:%Y-%m-%d}" for t in test_t],
        "actual": [int(v) for v in actual],
        "predicted": [round(float(v), 1) for v in predicted],
        "lastWeek": [int(v) for v in series[test_t - 7]],
    },
    "lastWindow": {
        "dates": [f"{d:%Y-%m-%d}" for d in dates.iloc[-args.window :]],
        "values": [int(v) for v in last_values],
    },
    "nextPrediction": round(next_prediction, 1),
}
info_path = PUBLIC / "ticket-rnn.json"
info_path.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")

print(f"Prévision pour le {dates.iloc[-1] + pd.Timedelta(days=1):%Y-%m-%d} : {next_prediction:.0f} tickets")
for path in (pth_path, onnx_path, info_path):
    print(f"Écrit : {path.relative_to(HERE.parent)}")
