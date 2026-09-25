"""Entraîne un VAE convolutif à débruiter des chiffres MNIST, puis l'exporte en ONNX."""

import argparse
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

# Plages de bruit : doivent rester identiques à celles de src/noise.ts.
SIZE = 28
GAUSS_MAX = 0.6
SALT_PEPPER_MAX = 0.3
MAX_BLOCKS = 3
BLOCK_MIN, BLOCK_MAX = 6, 10
NOISE_PROBABILITY = 0.5

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--latent", type=int, default=32)
parser.add_argument("--beta", type=float, default=0.5, help="poids du terme KL")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Périphérique : {device}")


def add_noise(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Gaussien, puis poivre et sel, puis occlusion ; chaque type actif une fois sur deux, par image."""
    b = x.shape[0]

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=g)

    def active() -> torch.Tensor:
        return rand(b, 1, 1, 1) < NOISE_PROBABILITY

    sigma = rand(b, 1, 1, 1) * GAUSS_MAX * active()
    x = (x + sigma * torch.randn(x.shape, generator=g)).clamp(0, 1)

    rate = rand(b, 1, 1, 1) * SALT_PEPPER_MAX * active()
    hit = rand(*x.shape) < rate
    salt = (rand(*x.shape) < 0.5).float()
    x = torch.where(hit, salt, x)

    blocks = torch.randint(1, MAX_BLOCKS + 1, (b,), generator=g) * active().view(b)
    coords = torch.arange(SIZE)
    for k in range(MAX_BLOCKS):
        side = torch.randint(BLOCK_MIN, BLOCK_MAX + 1, (b,), generator=g)
        x0 = (rand(b) * (SIZE - side + 1)).long()
        y0 = (rand(b) * (SIZE - side + 1)).long()
        cols = (coords >= x0[:, None]) & (coords < (x0 + side)[:, None])
        rows = (coords >= y0[:, None]) & (coords < (y0 + side)[:, None])
        mask = rows[:, :, None] & cols[:, None, :] & (k < blocks)[:, None, None]
        x = x.masked_fill(mask[:, None], 0.0)
    return x


class ConvVAE(nn.Module):
    """Image bruitée (B, 1, 28, 28) -> logits de l'image propre, μ et log σ² du latent."""

    def __init__(self, latent: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 14×14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 7×7
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(64 * 7 * 7, latent)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent)
        self.decoder = nn.Sequential(
            nn.Linear(latent, 64 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 14×14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 28×28
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        # En évaluation, z = μ : sortie déterministe et plus nette.
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decoder(z), mu, logvar


class Denoiser(nn.Module):
    """Modèle exporté : image bruitée -> image débruitée dans [0, 1]."""

    def __init__(self, vae: ConvVAE):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        return torch.sigmoid(self.vae(x)[0])


def losses(model: ConvVAE, noisy: torch.Tensor, clean: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits, mu, logvar = model(noisy)
    recon = F.binary_cross_entropy_with_logits(logits, clean, reduction="sum") / len(clean)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(clean)
    return recon, kl


# --- Données -------------------------------------------------------------------------
def load(train: bool) -> torch.Tensor:
    dataset = datasets.MNIST(root=str(DATA), train=train, download=True)
    return dataset.data.unsqueeze(1).float() / 255


X_train, X_test = load(True), load(False)
noise_gen = torch.Generator().manual_seed(args.seed)
# Bruit de test tiré une fois : les pertes restent comparables d'une époque à l'autre.
noisy_test = add_noise(X_test, torch.Generator().manual_seed(args.seed + 1))
print(f"Entraînement : {len(X_train)} images, test : {len(X_test)} images")

# --- Entraînement --------------------------------------------------------------------
model = ConvVAE(args.latent).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
params = f"{sum(p.numel() for p in model.parameters()):_}".replace("_", " ")
print(f"VAE convolutif : latent {args.latent}, β = {args.beta}, {params} paramètres")


def evaluate() -> tuple[float, float]:
    model.eval()
    recon_sum = kl_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(X_test), 1000):
            recon, kl = losses(model, noisy_test[i : i + 1000].to(device), X_test[i : i + 1000].to(device))
            n = len(X_test[i : i + 1000])
            recon_sum += recon.item() * n
            kl_sum += kl.item() * n
    return recon_sum / len(X_test), kl_sum / len(X_test)


print(f"\n{'Époque':>6}{'Recon.':>10}{'KL':>8}{'Recon. test':>13}{'KL test':>9}{'Durée':>8}")
start = time.perf_counter()
for epoch in range(1, args.epochs + 1):
    epoch_start = time.perf_counter()
    model.train()
    recon_sum = kl_sum = 0.0
    for idx in torch.randperm(len(X_train)).split(args.batch):
        clean = X_train[idx]
        noisy = add_noise(clean, noise_gen)
        recon, kl = losses(model, noisy.to(device), clean.to(device))
        optimizer.zero_grad()
        (recon + args.beta * kl).backward()
        optimizer.step()
        recon_sum += recon.item() * len(idx)
        kl_sum += kl.item() * len(idx)
    test_recon, test_kl = evaluate()
    print(
        f"{epoch:6d}{recon_sum / len(X_train):10.2f}{kl_sum / len(X_train):8.2f}"
        f"{test_recon:13.2f}{test_kl:9.2f}{time.perf_counter() - epoch_start:7.1f}s"
    )
print(f"Durée totale : {time.perf_counter() - start:.0f} s")

# --- Sauvegardes ---------------------------------------------------------------------
model = model.cpu().eval()
pth_path = HERE / "vae-denoiser.pth"
torch.save({"state_dict": model.state_dict(), "latent": args.latent, "beta": args.beta}, pth_path)

PUBLIC.mkdir(parents=True, exist_ok=True)
onnx_path = PUBLIC / "vae-denoiser.onnx"
denoiser = Denoiser(model).eval()
torch.onnx.export(
    denoiser,
    (torch.zeros(1, 1, SIZE, SIZE),),
    str(onnx_path),
    input_names=["input"],
    output_names=["output"],
    external_data=False,
)

# Vérifie que le modèle ONNX débruite comme PyTorch.
session = ort.InferenceSession(str(onnx_path))
samples = noisy_test[:64]
with torch.no_grad():
    expected = denoiser(samples).numpy()
actual = np.concatenate([session.run(None, {"input": x[None].numpy()})[0] for x in samples])
print(f"\nÉcart max PyTorch / ONNX : {np.abs(actual - expected).max():.2e}")

for path in (pth_path, onnx_path):
    print(f"Écrit : {path.relative_to(HERE.parent)}")
