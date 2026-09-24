# TP2 — Reconnaissance de chiffres manuscrits (Angular + ONNX)

## Contexte

Le script Python [main.py](main.py) entraîne un perceptron multicouche (MLP) sur
le jeu de données MNIST avec PyTorch, puis exporte le modèle entraîné au format
**ONNX** (`model_mnist.onnx`, présent à la racine du workspace). L'objectif de ce
TP est de **tester ce modèle dans un navigateur** : offrir une interface où un
utilisateur dessine un chiffre à la main et voit en direct ce que le réseau
reconnaît.

Le workspace `tps-laval` regroupe plusieurs TP. Le TP1 (dossier `tp1/`) est une
application web TypeScript/Vite lancée via `bun`/`mise`. Ce TP2 suit la même
logique d'organisation mais utilise **Angular**.

## Objectif

Construire une application web **Angular** qui :

1. Charge le modèle `model_mnist.onnx` et l'exécute côté navigateur.
2. Fournit une zone de dessin (canvas) où l'utilisateur trace un chiffre à la
   souris.
3. Affiche **en temps réel** le chiffre reconnu ainsi que la confiance du
   modèle pour chacune des 10 classes.

## Périmètre fonctionnel

### Chargement et inférence du modèle

- Le modèle `model_mnist.onnx` est **inclus dans le projet Angular** (copié dans
  les assets, servi statiquement) et chargé au démarrage de l'application.
- L'inférence s'exécute **côté client avec `onnxruntime-web` en backend WASM**.
- Entrée attendue par le modèle : un tenseur `1×1×28×28` de valeurs flottantes
  en niveaux de gris **dans l'intervalle 0–1** (cohérent avec `transforms.ToTensor()`).
- Sortie du modèle : un vecteur de **10 logits bruts** (aucun softmax n'est
  appliqué dans le réseau — voir `MLP.forward` dans [main.py](main.py)).

### Canvas de dessin

- Zone de dessin où l'utilisateur écrit un chiffre **à la souris**.
- **Cible : desktop uniquement** (interaction souris). Le support tactile
  mobile/tablette n'est pas requis.
- Couleurs du canvas d'affichage : **fond blanc, trait noir**.
  - ⚠️ MNIST utilise la convention inverse (fond noir, trait blanc). Le
    prétraitement doit donc **inverser les niveaux de gris** avant de construire
    le tenseur d'entrée, afin de correspondre au format d'entraînement.
- Bouton **« Effacer »** pour réinitialiser le canvas.

### Prétraitement de l'image

- Redimensionner le dessin en **28×28** en niveaux de gris.
- Inverser les couleurs (blanc→noir) pour respecter la convention MNIST.
- Normaliser les valeurs dans l'intervalle **0–1**.
- Afficher un **aperçu de l'image 28×28** effectivement envoyée au modèle (pour
  visualiser ce que « voit » le réseau).

### Affichage du résultat

- Déclenchement **automatique en temps réel** : la prédiction se met à jour
  pendant/après le tracé, sans action manuelle de l'utilisateur.
- Afficher **le chiffre prédit** (grand, mis en évidence).
- Afficher **la probabilité de chacune des 10 classes** (0 à 9), obtenue en
  appliquant un **softmax** sur les logits de sortie — par exemple sous forme
  d'histogramme ou de liste triée.

## Hors périmètre

- Support tactile mobile/tablette.
- Ré-entraînement ou modification du modèle depuis le navigateur.
- Import d'un fichier `.onnx` choisi par l'utilisateur (le modèle est figé et
  embarqué dans le projet).
- Sélection réglable de l'épaisseur du trait.
- Déploiement en ligne (hébergement public).

## Utilisateurs cibles

Public pédagogique du TP (étudiants/enseignant) souhaitant **vérifier
visuellement** la qualité du MLP entraîné sur MNIST, sur poste de travail
desktop.

## Contraintes techniques

- **Framework** : Angular (nouveau projet).
- **Emplacement du projet** : dans le dossier **`tp2/`**, mais les dépendances
  (`node_modules`) restent **à la racine du workspace** (`tps-laval/`) — les
  dépendances sont donc mutualisées/hoistées au niveau du workspace, cohérent
  avec l'organisation `bun`/`mise` existante.
- **Moteur d'inférence** : `onnxruntime-web` (backend **WASM**).
- **Exécution du modèle** : entièrement **côté client**, aucun backend serveur
  d'inférence.
- **Modèle** : `model_mnist.onnx`, entrée `1×1×28×28`, sortie 10 logits.

## Livraison / Exécution

- Le site doit pouvoir être **construit en version statique** (dossier `dist/`)
  via le build Angular.
- L'exécution locale via serveur de développement reste possible pendant le
  développement, mais **le livrable attendu est le build statique**.

## Tests

- **Validation manuelle** suffisante : dessiner plusieurs chiffres et vérifier
  que le modèle les reconnaît correctement.
- Pas de suite de tests unitaires ni de stratégie de non-régression exigée pour
  ce TP.

## Critères de succès

1. L'application Angular démarre et charge `model_mnist.onnx` sans erreur.
2. L'utilisateur peut dessiner un chiffre à la souris sur le canvas.
3. La prédiction s'affiche **automatiquement et en temps réel**.
4. Le chiffre reconnu et les probabilités des 10 classes sont affichés.
5. Un aperçu 28×28 de l'image envoyée au modèle est visible.
6. Le bouton « Effacer » réinitialise le canvas et la prédiction.
7. Des chiffres clairement tracés sont majoritairement bien reconnus
   (validation manuelle).
8. Un build statique dans `dist/` est produit avec succès.

## Points en suspens

- Aucun point bloquant. Le rendu exact de l'affichage des probabilités
  (histogramme vs liste) est laissé à l'appréciation lors de l'implémentation.
