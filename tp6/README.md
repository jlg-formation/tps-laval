# TP6 — Explorer l'espace latent 2D d'un VAE MNIST

Sujet : [brief.md](brief.md).

## Chaîne complète

```bash
mise run tp6-train   # train.py -> vae-mnist.pth, public/decoder.onnx, public/latent-points.json
mise run tp6         # application Web : http://localhost:5173
```

| Fichier | Rôle |
| --- | --- |
| `train.py` | VAE MLP (784 → 256 → 64 → μ, log σ² en 2D ; décodeur 2 → 64 → 256 → 784), perte BCE + β·KL, export du décodeur en ONNX (batch dynamique) avec vérification de parité, export de 2 000 μ du test (200 par chiffre) et des bornes du plan. CUDA si disponible, sinon CPU. |
| `src/decoder.ts` | Chargement de `decoder.onnx` : `latent` [batch, 2] → `output` [batch, 1, 28, 28]. |
| `src/latent.ts` | Format de `latent-points.json`, palette des chiffres. |
| `src/plane.ts` | Plan latent : fond (nuage ou grille), point déplaçable à la souris, au doigt ou au clavier. |
| `src/main.ts` | Décodage en temps réel du point, grille 15 × 15 de chiffres décodés, légende. |

## Expériences

```bash
mise run tp6-train --beta 0.1    # latent moins contraint : régions plus étalées, trous entre elles ?
mise run tp6-train --beta 4      # latent très régulier : chiffres plus flous ?
mise run tp6-train --epochs 50
```

## Questions de discussion

- Les dix chiffres forment-ils dix régions séparées ? Lesquels se chevauchent, et pourquoi ?
- Que génère le décodeur loin de tous les points du test ?
- Les transitions entre chiffres sont-elles continues ? Par quelles formes passe-t-on de 1 à 7 ?
- Pourquoi utiliser μ plutôt qu'un z échantillonné pour placer les images du test ?
- Pourquoi la grille décodée ressemble-t-elle au nuage coloré ?
