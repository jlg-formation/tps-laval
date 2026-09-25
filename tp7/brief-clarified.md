# TP7 — GAN sur MNIST

## Contexte

TP de formation pour des stagiaires qui ont déjà suivi les TP précédents du dépôt (MLP, CNN sur MNIST). Ils connaissent la boucle d'entraînement PyTorch, mais découvrent les réseaux antagonistes génératifs (GAN). Beaucoup n'ont qu'un **CPU** : l'entraînement doit rester court.

## Objectif

Entraîner un GAN sur MNIST :

- **Discriminateur (D)** : reçoit une image 28×28 et donne la probabilité qu'elle soit une vraie image MNIST plutôt qu'une image générée.
- **Générateur (G)** : transforme un bruit aléatoire $z$ en une image qui ressemble à un chiffre MNIST, pour tromper D.

On peut ensuite générer des chiffres dans une page web, à partir du générateur exporté en ONNX.

## Périmètre fonctionnel

### Architecture du projet

L'architecture du tp7 doit rester **dans le même esprit que les TP précédents** (en particulier tp5 et tp6), pour que les stagiaires s'y retrouvent immédiatement :

- Même organisation de dossier :

  ```text
  tp7/
    brief.md
    README.md
    train.py            # entraînement + export ONNX
    index.html
    vite.config.ts
    scripts/copy-ort.ts # copie des fichiers wasm d'onnxruntime-web dans public/ort/
    public/             # generator.onnx, ort/
    src/                # main.ts, style.css + un module par responsabilité (ex. generator.ts, grid.ts)
  ```

- Mêmes conventions de code : constantes de chemins en tête de `train.py` (`HERE`, `DATA = HERE.parent / "data"`, `PUBLIC = HERE / "public"`), options en ligne de commande (`--epochs`, `--batch`, `--lr`, `--latent`, `--seed`), style et niveau de commentaires identiques.
- Mêmes scripts dans le `package.json` racine : `tp7:ort`, `tp7:dev`, `tp7:build`, calqués sur ceux de tp6.
- Mêmes tâches `mise` : `tp7-train` et `tp7`, avec description et `env = { PYTHONIOENCODING = "utf-8" }`.
- Même style de page web (structure HTML, CSS) que les TP précédents.

### Entraînement (Python)

- Script `tp7/train.py`, lancé par `mise run tp7-train` (tâche à ajouter dans `mise.toml` avec `PYTHONIOENCODING = "utf-8"`, comme `tp6-train`).
- Stack : PyTorch, dépendances gérées par `uv`, comme les TP précédents.
- Architecture : GAN « vanilla » avec des MLP, pensé pour le CPU.
  - Générateur : $z \in \mathbb{R}^{64}$ → Linear 256 → LeakyReLU → Linear 512 → LeakyReLU → Linear 784 → `tanh`.
  - Discriminateur : 784 → Linear 512 → LeakyReLU → Linear 256 → LeakyReLU → Linear 1 (logits, avec `BCEWithLogitsLoss`).
  - Images normalisées dans $[-1, 1]$.
  - Optimiseur Adam (lr = 2e-4, betas = (0.5, 0.999)), batch de 128, environ 20 époques.
- Données MNIST : réutiliser le dossier partagé `data/` à la racine du dépôt (`HERE.parent / "data"`, comme tp5 et tp6). S'il est absent, MNIST est téléchargé automatiquement (`torchvision.datasets.MNIST(download=True)`) au même endroit.
- Suivi de l'entraînement :
  - Logs dans la console à chaque époque : perte de D, perte de G, moyenne de D(x) et de D(G(z)).
  - Une grille PNG d'images générées à chaque époque, dans `tp7/samples/`, à partir d'un bruit $z$ fixe pour voir l'évolution. **`tp7/samples/` doit être ajouté au `.gitignore`.**
- En fin d'entraînement : export du **générateur seul** en ONNX, dans `tp7/public/generator.onnx` (entrée `[batch, 64]`, sortie `[batch, 1, 28, 28]`). Le fichier est déjà ignoré par git grâce à la règle `*.onnx`.

### Visualisation (web)

- Application Vite + TypeScript dans `tp7/`, avec inférence par `onnxruntime-web` (même approche que tp4 à tp6 : script `copy-ort.ts`, dossier `tp7/public/ort/` à ajouter au `.gitignore`).
- Lancement par `mise run tp7` (script `tp7:dev` dans `package.json`).
- Page simple : une grille d'images générées à partir de bruit aléatoire, et un bouton **« Générer »** qui tire un nouveau $z$ et met la grille à jour.
- Pas de matplotlib : toute la visualisation interactive se fait dans le web.

### Documentation

`tp7/README.md` contient :

- une explication intuitive du GAN (le faussaire G contre le policier D) ;
- un schéma Mermaid du flux : bruit $z$ → G → image fausse → D ← image réelle MNIST → vrai/faux ;
- les commandes à lancer (`mise run tp7-train`, puis `mise run tp7`).

## Hors périmètre

- DCGAN, GAN conditionnel, WGAN : possibles en extension plus tard, pas dans ce TP.
- Curseurs sur l'espace latent, interpolation, mode « jeu vrai/faux », courbes de pertes dans la page web.
- Modèle pré-entraîné versionné : chaque stagiaire entraîne le sien.
- Code à trous, énoncé d'exercices : le code est fourni **complet et commenté**. Les stagiaires lancent, observent et modifient les hyperparamètres.
- Formules mathématiques détaillées (minimax) et métriques automatiques (FID, classification par un CNN).
- TensorBoard.

## Utilisateurs cibles

Stagiaires en formation, niveau intermédiaire (ils ont suivi les TP MLP et CNN), la plupart sur des machines **sans GPU**, sous Windows ou autre, avec le dépôt cloné et préparé par `mise run prepare`.

## Contraintes techniques

- Python + PyTorch (`uv`), front Vite/TypeScript (`bun`), orchestration par `mise`, cohérents avec les TP existants.
- Entraînement complet raisonnable sur CPU (quelques minutes).
- Le générateur seul est exporté en ONNX. Le discriminateur ne sert qu'à l'entraînement.
- Fichiers générés hors git : `data/`, `tp7/samples/`, `*.onnx`, `tp7/public/ort/`.
- Accès Internet nécessaire au premier lancement si MNIST n'est pas encore présent dans `data/`.

## Critères de succès

- `mise run tp7-train` se termine sans erreur sur une machine CPU en quelques minutes.
- À la fin, la plupart des chiffres générés (dans `samples/` et dans la page web) sont **reconnaissables à l'œil**.
- `mise run tp7` ouvre une page où le bouton « Générer » affiche une nouvelle grille de chiffres à chaque clic.
- Un stagiaire peut lancer et comprendre le TP en s'aidant seulement du `README`.
