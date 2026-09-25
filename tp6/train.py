"""Entraîne un VAE MLP à espace latent 2D sur MNIST, exporte le décodeur en ONNX et les points latents du test."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
PUBLIC = HERE / "public"

SIZE = 28
LATENT = 2
POINTS_PER_CLASS = 200
BOUNDS_MARGIN = 0.1

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--beta", type=float, default=1.0, help="poids du terme KL")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Périphérique : {device}")


class Encoder(nn.Module):
    """Image (B, 1, 28, 28) -> μ et log σ² du latent (B, 2)."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(SIZE * SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, LATENT)
        self.fc_logvar = nn.Linear(64, LATENT)

    def forward(self, x):
        h = self.body(x)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """Latent (B, 2) -> image (B, 1, 28, 28) dans [0, 1]."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(LATENT, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, SIZE * SIZE),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, SIZE, SIZE)),
        )

    def forward(self, latent):
        return self.body(latent)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """z = μ + σ ⊙ ε, avec ε ~ N(0, I) et σ = exp(½ log σ²)."""
    return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        return self.decoder(reparameterize(mu, logvar)), mu, logvar


def losses(model: VAE, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reconstruction (BCE) et KL, sommées sur les pixels / dimensions, moyennées sur le batch."""
    x_hat, mu, logvar = model(x)
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(x)
    return recon, kl


# --- Données -------------------------------------------------------------------------
def load(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.MNIST(root=str(DATA), train=train, download=True)
    return dataset.data.unsqueeze(1).float() / 255, dataset.targets


X_train, _ = load(True)
X_test, y_test = load(False)
print(f"Entraînement : {len(X_train)} images, test : {len(X_test)} images")

# --- Entraînement --------------------------------------------------------------------
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
params = f"{sum(p.numel() for p in model.parameters()):_}".replace("_", " ")
print(f"VAE MLP : latent {LATENT}, β = {args.beta}, {params} paramètres")


def evaluate() -> tuple[float, float]:
    model.eval()
    recon_sum = kl_sum = 0.0
    with torch.no_grad():
        for x in X_test.split(1000):
            recon, kl = losses(model, x.to(device))
            recon_sum += recon.item() * len(x)
            kl_sum += kl.item() * len(x)
    return recon_sum / len(X_test), kl_sum / len(X_test)


print(f"\n{'Époque':>6}{'Total':>9}{'Recon.':>9}{'KL':>7}{'Total test':>12}{'Recon. test':>13}{'KL test':>9}{'Durée':>8}")
start = time.perf_counter()
for epoch in range(1, args.epochs + 1):
    epoch_start = time.perf_counter()
    model.train()
    recon_sum = kl_sum = 0.0
    for idx in torch.randperm(len(X_train)).split(args.batch):
        x = X_train[idx].to(device)
        recon, kl = losses(model, x)
        optimizer.zero_grad()
        (recon + args.beta * kl).backward()
        optimizer.step()
        recon_sum += recon.item() * len(idx)
        kl_sum += kl.item() * len(idx)
    recon_train, kl_train = recon_sum / len(X_train), kl_sum / len(X_train)
    recon_test, kl_test = evaluate()
    print(
        f"{epoch:6d}{recon_train + args.beta * kl_train:9.2f}{recon_train:9.2f}{kl_train:7.2f}"
        f"{recon_test + args.beta * kl_test:12.2f}{recon_test:13.2f}{kl_test:9.2f}"
        f"{time.perf_counter() - epoch_start:7.1f}s"
    )
print(f"Durée totale : {time.perf_counter() - start:.0f} s")

# --- Sauvegardes ---------------------------------------------------------------------
model = model.cpu().eval()
pth_path = HERE / "vae-mnist.pth"
torch.save({"state_dict": model.state_dict(), "beta": args.beta, "epochs": args.epochs}, pth_path)

PUBLIC.mkdir(parents=True, exist_ok=True)
onnx_path = PUBLIC / "decoder.onnx"
torch.onnx.export(
    model.decoder,
    (torch.zeros(1, LATENT),),
    str(onnx_path),
    input_names=["latent"],
    output_names=["output"],
    dynamic_shapes=({0: torch.export.Dim("batch")},),
    external_data=False,
)

# Vérifie que le décodeur ONNX produit les mêmes images que PyTorch, pour un et plusieurs points.
session = ort.InferenceSession(str(onnx_path))
for n in (1, 225):
    z = torch.randn(n, LATENT) * 2
    with torch.no_grad():
        expected = model.decoder(z).numpy()
    (actual,) = session.run(None, {"latent": z.numpy()})
    print(f"Écart max PyTorch / ONNX (batch {n}) : {np.abs(actual - expected).max():.2e}, forme {actual.shape}")

# μ de tout le test : les bornes couvrent l'ensemble, les points affichés sont un échantillon équilibré.
with torch.no_grad():
    mu = torch.cat([model.encoder(x)[0] for x in X_test.split(1000)])
lo, hi = mu.min(0).values, mu.max(0).values
margin = (hi - lo) * BOUNDS_MARGIN
lo, hi = lo - margin, hi + margin

g = torch.Generator().manual_seed(args.seed)
picked = torch.cat(
    [(y_test == d).nonzero().view(-1)[torch.randperm(int((y_test == d).sum()), generator=g)[:POINTS_PER_CLASS]] for d in range(10)]
)
# Mélangés pour qu'aucune classe ne soit systématiquement dessinée par-dessus les autres.
picked = picked[torch.randperm(len(picked), generator=g)]
points_path = PUBLIC / "latent-points.json"
payload = {
    "bounds": {"z1": [round(lo[0].item(), 3), round(hi[0].item(), 3)], "z2": [round(lo[1].item(), 3), round(hi[1].item(), 3)]},
    "points": [[round(mu[i, 0].item(), 3), round(mu[i, 1].item(), 3), int(y_test[i])] for i in picked.tolist()],
}
points_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
print(f"Points latents : {len(payload['points'])}, bornes z1 {payload['bounds']['z1']}, z2 {payload['bounds']['z2']}")

for path in (pth_path, onnx_path, points_path):
    print(f"Écrit : {path.relative_to(HERE.parent)}")
