# TP5 — VAE convolutif débruiteur sur MNIST

## Contexte

Cinquième TP de la série `tps-laval` (deep learning avec Python et PyTorch). Après le perceptron (TP1), le MLP et le CNN MNIST (TP2/TP3) et le RNN de prévision (TP4), ce TP introduit un **VAE** (*Variational Autoencoder*) utilisé comme **débruiteur** d'images de chiffres manuscrits MNIST.

Le dataset MNIST est déjà présent dans `data/MNIST/raw/`.

## Objectif

Entraîner un VAE qui prend en entrée un chiffre MNIST **bruité** et restitue le chiffre **propre**. L'objectif est avant tout la **qualité du débruitage** : les images reconstruites doivent être lisibles et fidèles à l'original, y compris sous des bruits variés et assez forts.

Le résultat est présenté dans une **démo web** qui exécute le modèle exporté en ONNX dans le navigateur.

## Périmètre fonctionnel

### 1. Bruitage des données

- Plusieurs types de bruit, **combinables**, appliqués dans cet ordre :

  | Type | Paramètre | Plage (entraînement et démo) |
  |---|---|---|
  | **Gaussien additif** | écart-type σ | 0 – 0,6 |
  | **Poivre et sel** | proportion p de pixels forcés à 0 ou 1 (moitié/moitié) | 0 – 30 % |
  | **Occlusion** | nombre de blocs carrés mis à 0, côté tiré dans 6 – 10 px, position aléatoire | 0 – 3 blocs |

- Pendant l'entraînement, le bruit est tiré **aléatoirement à la volée, image par image** :
  - chaque type est activé indépendamment avec une probabilité de 0,5 ;
  - s'il est actif, son intensité est tirée uniformément dans la plage ci-dessus ;
  - environ 10 % des images restent sans aucun bruit, pour que le modèle apprenne aussi à ne pas abîmer une image propre.
- Le modèle ne voit donc jamais deux fois exactement la même image bruitée.
- Les valeurs de pixels restent dans [0, 1] après bruitage (clamp).
- La démo utilise les mêmes plages que l'entraînement, afin que le modèle ne reçoive pas de bruit plus fort que ce qu'il a appris.

### 2. Modèle : VAE convolutif débruiteur

- **Entrée** : image bruitée `[B, 1, 28, 28]`, valeurs dans [0, 1].
- **Cible** : image propre correspondante.
- **Encodeur** convolutif (Conv2d + activation, avec downsampling) qui produit `μ` et `log σ²` d'un espace latent de **dimension 32**.
- **Reparamétrisation** : `z = μ + σ · ε`, avec `ε ~ N(0, I)`.
- **Décodeur** convolutif (ConvTranspose2d ou upsampling + Conv2d) qui produit une image `[B, 1, 28, 28]` avec une sortie sigmoïde.
- **Perte** = `BCE(sortie, image propre) + β · KL(q(z|x) ‖ N(0, I))` :
  - la **BCE** est sommée sur les 784 pixels puis moyennée sur le batch. Elle est adaptée à des pixels dans [0, 1] et donne des traits plus nets que la MSE ;
  - la **KL** est sommée sur les 32 dimensions latentes puis moyennée sur le batch ;
  - **β = 0,5 par défaut** (option `--beta`). C'est un compromis : on réduit le flou dû au terme KL tout en gardant un vrai VAE (avec β = 1, les reconstructions sont plus floues).
- **En inférence et à l'export**, on utilise `z = μ` (sans échantillonnage) pour obtenir une sortie déterministe et nette.

### 3. Entraînement : `tp5/train.py`

- Chargement de MNIST depuis `data/MNIST` (torchvision).
- Bruitage aléatoire à la volée, comme décrit plus haut.
- Affichage des pertes (reconstruction et KL) par époque, sur train et test.
- Matériel : CUDA est utilisé **s'il est disponible**, sinon le CPU. Le périphérique est affiché au démarrage. Le **CPU est la cible de référence** : les stagiaires n'ont pas forcément de GPU, et le TP doit rester probant sans.
- Hyperparamètres exposés en options CLI, dans l'esprit de TP4. Valeurs par défaut :

  | Option | Défaut |
  |---|---|
  | `--epochs` | 10 |
  | `--batch` | 128 |
  | `--lr` | 1e-3 (Adam) |
  | `--latent` | 32 |
  | `--beta` | 0,5 |

- Ces valeurs sont dimensionnées pour le CPU. Le VAE convolutif reste petit (quelques centaines de milliers de paramètres au plus), pour que l'entraînement complet se termine en quelques minutes sur un portable standard. Il doit tout de même produire des chiffres débruités nets. Aucune qualité supplémentaire ne doit dépendre de CUDA.
- Graine aléatoire fixée pour la reproductibilité.
- Sortie : `tp5/public/vae-denoiser.onnx`, le modèle d'inférence. Entrée `[1, 1, 28, 28]` bruitée, sortie `[1, 1, 28, 28]` débruitée, en passant par encodeur → μ → décodeur.

### 4. Démo web (Vite + TypeScript vanilla, comme TP4)

- Chargement du modèle ONNX avec `onnxruntime-web` (wasm copié dans `tp5/public/ort/` par un script `copy-ort.ts`, comme TP4).
- **Dessin** : l'utilisateur dessine un chiffre à la souris ou au doigt (pointer events) dans un canvas, trait noir sur fond blanc. Un bouton **Effacer** vide le canvas.
- **Prétraitement MNIST** du dessin, repris de TP2 ([tp2/src/app/preprocess.ts](../tp2/src/app/preprocess.ts)) :
  - passage en trait blanc sur fond noir ;
  - réduction pour tenir dans une boîte de 20×20 ;
  - centrage par centre de masse dans une image 28×28, valeurs dans [0, 1].
- **Ajout de bruit par boutons**, un par type. Chaque clic ajoute une dose de bruit, cumulable et plafonnée au maximum vu à l'entraînement :
  - **+ Gaussien** : σ += 0,15, jusqu'à 0,6 ;
  - **+ Poivre et sel** : p += 7,5 %, jusqu'à 30 % ;
  - **+ Occlusion** : un bloc de plus, jusqu'à 3 ;
  - **Retirer le bruit** : remet toutes les intensités à 0.
- Le niveau courant de chaque bruit est affiché à côté de son bouton.
- La **réalisation du bruit est figée** : le champ gaussien, le tirage des pixels poivre et sel et la position et la taille des blocs sont tirés une fois, puis seulement mis à l'échelle par l'intensité. Quand l'utilisateur continue à dessiner, le bruit ne scintille donc pas. De nouveaux tirages ont lieu à chaque clic sur un bouton de bruit et à chaque **Effacer**.
- Le bruit est appliqué **côté navigateur**, en TypeScript, avec la même logique que l'entraînement.
- **Débruitage en temps réel** : l'inférence est relancée à chaque modification, pendant le tracé comme après chaque clic sur un bouton de bruit. Elle est limitée à une inférence en cours à la fois et calée sur `requestAnimationFrame`, pour ne pas saturer le navigateur.
- **Affichage côte à côte**, agrandi en pixels nets (`image-rendering: pixelated`) :
  - le dessin prétraité 28×28 (propre) ;
  - la version bruitée ;
  - la version débruitée.

### 5. Commandes (`mise.toml` et `package.json` racine)

- `mise run tp5-train` → `uv run tp5/train.py` (avec `PYTHONIOENCODING=utf-8`).
- `mise run tp5` → `bun run tp5:dev` (serveur Vite, root `tp5`, build vers `tp5/dist`).

## Hors périmètre

- Métriques quantitatives (PSNR, SSIM, précision d'un classifieur sur les images débruitées) : l'évaluation est **visuelle uniquement**.
- Comparaison avec un autoencodeur débruiteur classique (DAE) ou un β-VAE.
- Tirage de chiffres du jeu de test MNIST dans la démo : seul le dessin sert d'entrée.
- Sliders d'intensité : le bruit se règle uniquement avec les boutons.
- Exploration ou génération à partir de l'espace latent dans la démo.
- Notebook Jupyter.
- Déploiement en production, CI/CD, monitoring.

## Contraintes techniques

- **Python** avec PyTorch et torchvision, dépendances gérées par `uv` (`pyproject.toml` racine). Le CPU est la cible de référence, et CUDA sert d'accélération facultative.
- **JS/TS** : dépendances dans le `package.json` racine (bun), Vite vanilla TypeScript, `onnxruntime-web`.
- Export ONNX avec `external_data=False` (fichier unique) ; `*.onnx` est gitignoré.
- Sous Windows, `PYTHONIOENCODING=utf-8` est requis pour l'exporteur ONNX.
- La logique de bruitage doit être **équivalente** en Python (entraînement) et en TypeScript (démo).
- Conventions de structure alignées sur TP4 : `tp5/train.py`, `tp5/index.html`, `tp5/vite.config.ts`, `tp5/src/*.ts`, `tp5/scripts/copy-ort.ts`, `tp5/public/`.

## Critères de succès

- `mise run tp5-train` entraîne le modèle sans erreur **sur une machine sans GPU**, en quelques minutes, et produit le `.onnx`.
- `mise run tp5` lance une démo fonctionnelle :
  - dessin au canvas ;
  - boutons de bruit cumulables ;
  - image débruitée mise à jour sans délai perceptible pendant le tracé et après chaque clic ;
  - affichage propre / bruité / débruité.
- À l'œil, les chiffres **dessinés à la main** et débruités sont **nets et reconnaissables** pour des niveaux de bruit modérés à forts, sur chacun des trois types de bruit et sur leurs combinaisons.
