# TP5 — Débruitage de chiffres MNIST avec un VAE

Sujet : [brief.md](brief.md), précisé dans [brief-clarified.md](brief-clarified.md).

## Chaîne complète

```bash
mise run tp5-train   # train.py -> vae-denoiser.pth, public/vae-denoiser.onnx
mise run tp5         # application Web : http://localhost:5173
```

| Fichier | Rôle |
| --- | --- |
| `train.py` | Bruitage aléatoire à la volée (gaussien, poivre et sel, occlusion), VAE convolutif (latent 32), perte BCE + β·KL, export ONNX (encodeur → μ → décodeur) et vérification de parité. CUDA si disponible, sinon CPU. |
| `src/noise.ts` | Même bruitage qu'à l'entraînement, avec un tirage figé pour que le bruit ne scintille pas pendant le dessin. |
| `src/preprocess.ts` | Dessin -> format MNIST 28×28 (repris du TP2). |
| `src/main.ts` | Dessin, boutons de bruit, débruitage en temps réel. |

## Expériences

```bash
mise run tp5-train --beta 1      # VAE « standard » : reconstructions plus floues ?
mise run tp5-train --beta 0.1    # proche d'un autoencodeur débruiteur classique
mise run tp5-train --latent 2    # espace latent minuscule : que reste-t-il du chiffre ?
mise run tp5-train --epochs 30
```

## Questions de discussion

- Que se passe-t-il si le bruit dépasse ce que le modèle a vu à l'entraînement ?
- Pourquoi le VAE « invente »-t-il parfois un autre chiffre sous une forte occlusion ?
- Quel est le rôle du terme KL ? Que change β sur la netteté ?
- Pourquoi utiliser μ plutôt qu'un z échantillonné pour débruiter ?
- Un chiffre dessiné à la souris ressemble-t-il vraiment à un chiffre MNIST ?
