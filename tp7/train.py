"""Entraîne un GAN MLP sur MNIST, enregistre une grille d'échantillons par époque et exporte le générateur en ONNX."""

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.utils import save_image

HERE = Path(__file__).parent
DATA = HERE.parent / "data"
PUBLIC = HERE / "public"
SAMPLES = HERE / "samples"

SIZE = 28
GRID = 8

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--latent", type=int, default=64, help="dimension du bruit z")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Périphérique : {device}")


class Generator(nn.Module):
    """Bruit z (B, latent) -> image (B, 1, 28, 28) dans [-1, 1]."""

    def __init__(self, latent: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(latent, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, SIZE * SIZE),
            nn.Tanh(),
            nn.Unflatten(1, (1, SIZE, SIZE)),
        )

    def forward(self, z):
        return self.body(z)


class Discriminator(nn.Module):
    """Image (B, 1, 28, 28) -> logit (B, 1) : positif = « vraie image MNIST »."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(SIZE * SIZE, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.body(x)


# --- Données -------------------------------------------------------------------------
dataset = datasets.MNIST(root=str(DATA), train=True, download=True)
# Même échelle que la sortie tanh du générateur.
X_train = dataset.data.unsqueeze(1).float() / 127.5 - 1
print(f"Entraînement : {len(X_train)} images")

# --- Entraînement --------------------------------------------------------------------
G = Generator(args.latent).to(device)
D = Discriminator().to(device)
opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()


def count(model: nn.Module) -> str:
    return f"{sum(p.numel() for p in model.parameters()):_}".replace("_", " ")


print(f"GAN MLP : latent {args.latent}, générateur {count(G)} paramètres, discriminateur {count(D)} paramètres")

# Toujours le même bruit : les grilles successives montrent l'évolution du générateur.
fixed_z = torch.randn(GRID * GRID, args.latent, device=device)
SAMPLES.mkdir(parents=True, exist_ok=True)

print(f"\n{'Époque':>6}{'Perte D':>10}{'Perte G':>10}{'D(x)':>8}{'D(G(z))':>10}{'Durée':>8}")
start = time.perf_counter()
for epoch in range(1, args.epochs + 1):
    epoch_start = time.perf_counter()
    G.train()
    loss_d_sum = loss_g_sum = d_real_sum = d_fake_sum = 0.0
    for idx in torch.randperm(len(X_train)).split(args.batch):
        real = X_train[idx].to(device)
        ones = torch.ones(len(idx), 1, device=device)
        zeros = torch.zeros(len(idx), 1, device=device)
        fake = G(torch.randn(len(idx), args.latent, device=device))

        # D apprend à répondre 1 sur les vraies images et 0 sur les fausses ; detach() fige G.
        logits_real = D(real)
        logits_fake = D(fake.detach())
        loss_d = criterion(logits_real, ones) + criterion(logits_fake, zeros)
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()

        # G cherche à faire répondre 1 à D sur ses fausses images (perte « non saturante »).
        loss_g = criterion(D(fake), ones)
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        loss_d_sum += loss_d.item() * len(idx)
        loss_g_sum += loss_g.item() * len(idx)
        d_real_sum += torch.sigmoid(logits_real).sum().item()
        d_fake_sum += torch.sigmoid(logits_fake).sum().item()

    G.eval()
    with torch.no_grad():
        save_image(G(fixed_z), SAMPLES / f"epoch-{epoch:02d}.png", nrow=GRID, normalize=True, value_range=(-1, 1))
    n = len(X_train)
    print(
        f"{epoch:6d}{loss_d_sum / n:10.3f}{loss_g_sum / n:10.3f}{d_real_sum / n:8.3f}{d_fake_sum / n:10.3f}"
        f"{time.perf_counter() - epoch_start:7.1f}s"
    )
print(f"Durée totale : {time.perf_counter() - start:.0f} s")

# --- Sauvegardes ---------------------------------------------------------------------
G, D = G.cpu().eval(), D.cpu().eval()
pth_path = HERE / "gan-mnist.pth"
torch.save({"generator": G.state_dict(), "discriminator": D.state_dict(), "args": vars(args)}, pth_path)

PUBLIC.mkdir(parents=True, exist_ok=True)
onnx_path = PUBLIC / "generator.onnx"
torch.onnx.export(
    G,
    (torch.zeros(1, args.latent),),
    str(onnx_path),
    input_names=["latent"],
    output_names=["output"],
    dynamic_shapes=({0: torch.export.Dim("batch")},),
    external_data=False,
)

# Vérifie que le générateur ONNX produit les mêmes images que PyTorch, pour un et plusieurs bruits.
session = ort.InferenceSession(str(onnx_path))
for n in (1, GRID * GRID):
    z = torch.randn(n, args.latent)
    with torch.no_grad():
        expected = G(z).numpy()
    (actual,) = session.run(None, {"latent": z.numpy()})
    print(f"Écart max PyTorch / ONNX (batch {n}) : {np.abs(actual - expected).max():.2e}, forme {actual.shape}")

for path in (pth_path, onnx_path, SAMPLES):
    print(f"Écrit : {path.relative_to(HERE.parent)}")
