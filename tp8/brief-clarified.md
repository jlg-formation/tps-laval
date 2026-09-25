# TP 8 — Apprentissage par renforcement : un agent Q-learning dans un labyrinthe

## Contexte

Le TP 8 du cours porte sur l'apprentissage par renforcement (RL). Les TP 1 à 7 portaient sur l'apprentissage supervisé et les modèles génératifs (perceptron, MLP, CNN, RNN, VAE, GAN). Celui-ci présente le paradigme agent / environnement / récompense sur le problème classique du labyrinthe.

Le TP est un **site web autonome** : tout le RL tourne dans le navigateur. Pas de Python, pas de modèle ONNX, pas de fichier `.pth`.

## Objectif

Montrer en temps réel un agent qui apprend par **Q-learning tabulaire** à aller de la case de départ jusqu'à la sortie d'un labyrinthe généré aléatoirement. On doit voir la table Q se construire et la politique converger vers un chemin court.

## Périmètre fonctionnel

### 1. Environnement (labyrinthe)

- Une grille de cases. Chaque case est soit un **mur**, soit un **couloir**.
- Une case **départ** (S) fixée dans le coin haut-gauche et une case **sortie** (G) dans le coin bas-droit. Les deux coins sont opposés, donc le chemin est long et lisible, et d'un labyrinthe à l'autre on garde les mêmes repères.
- **Génération aléatoire** avec un bouton « Nouveau labyrinthe ». Le labyrinthe est « parfait » : il existe toujours un chemin de S à G (par exemple via un DFS / recursive backtracker).
- **Taille réglable** avec un curseur de 5 à 25 cellules de côté (pas de 1, **10 par défaut**). La grille affichée fait (2n+1)×(2n+1) cases, murs compris. Changer la taille génère un nouveau labyrinthe et réinitialise l'agent.
- Environnement **déterministe** : pas de pièges, pas de bonus, pas de sol glissant.
- Actions : 4 déplacements (haut, bas, gauche, droite). Aller dans un mur ou sortir de la grille laisse l'agent sur place.
- Un épisode se termine quand l'agent atteint G (**terminal**), ou quand il est **tronqué** au bout de `4 × nombre de cases couloir` pas. Cette limite laisse une vraie chance à la marche aléatoire des premiers épisodes sans bloquer l'interface. À la troncature, la mise à jour garde le terme $\gamma \max Q(s',\cdot)$, car s' n'est pas un état terminal : c'est la distinction terminaison / troncature. Sur la courbe, les épisodes tronqués ont une couleur à part.

### 2. Récompenses (réglables)

Valeurs par défaut, modifiables dans l'interface :

| Événement            | Récompense par défaut |
| -------------------- | --------------------- |
| Atteindre la sortie  | +100                  |
| Pas ordinaire        | −1                    |
| Heurter un mur       | −5                    |

### 3. Agent Q-learning tabulaire

- Table `Q[état][action]`, où un état = une case. Initialisée à 0.
- Politique d'exploration **ε-greedy**.
- Mise à jour :

  $$Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]$$

  (sans le terme $\gamma \max Q(s',\cdot)$ si $s'$ est terminal).
- **Décroissance de ε** à la fin de chaque épisode : $\varepsilon \leftarrow \max(\varepsilon_{min},\ \varepsilon \cdot \text{decay})$.

### 4. Hyperparamètres réglables dans l'interface

Chaque curseur affiche sa valeur et une infobulle d'une ligne sur son rôle.

| Paramètre            | Défaut | Plage         | Pas   |
| -------------------- | ------ | ------------- | ----- |
| $\alpha$             | 0,1    | 0,01 – 1      | 0,01  |
| $\gamma$             | 0,99   | 0,5 – 1       | 0,005 |
| $\varepsilon$ initial | 1      | 0 – 1         | 0,05  |
| $\varepsilon_{min}$  | 0,05   | 0 – 0,5       | 0,01  |
| Décroissance ε       | 0,99   | 0,9 – 0,999   | 0,001 |
| Récompense sortie    | +100   | 0 – 500       | 10    |
| Récompense pas       | −1     | −10 – 0       | 0,5   |
| Récompense mur       | −5     | −20 – 0       | 1     |

Pourquoi ces défauts :
- **γ = 0,99** plutôt que 0,95 : dans un 25×25, le chemin fait plusieurs centaines de pas. Avec 0,95, $\gamma^{200} \approx 0$ et la heatmap sature. Avec 0,99, le gradient depuis G reste visible.
- **Décroissance 0,99** : ε passe de 1 à 0,05 en environ 300 épisodes. La transition exploration → exploitation se voit sur un 10×10 sans y passer la séance.
- **α = 0,1** : valeur classique du cours, assez stable. L'environnement est déterministe, donc on peut monter α jusqu'à 1 : c'est une expérience à suggérer.
- Un bouton **« Valeurs par défaut »** remet tous les curseurs à ces valeurs.

Modifier un hyperparamètre pendant l'entraînement l'applique immédiatement, sans reset. La valeur courante de ε s'affiche en continu.

### 5. Contrôles de l'entraînement

- **Play / Pause** : lance ou suspend l'apprentissage en continu.
- **Pas à pas** : exécute une seule transition (s, a, r, s'), utile pour suivre une mise à jour de Q.
- **Vitesse réglable** : nombre de pas par frame d'animation.
- **Mode turbo** : entraîne le plus vite possible (plusieurs épisodes par frame, sans animer l'agent) et rafraîchit la visualisation de temps en temps.
- **Reset** : remet Q à zéro et ε à sa valeur initiale, en gardant le même labyrinthe.

### 6. Visualisations

- **Grille** (Canvas) : murs, S, G et position courante de l'agent.
- **Heatmap de $V(s) = \max_a Q(s,a)$** en fond des cases couloir, avec une échelle de couleur.
- **Flèches de politique** : $\arg\max_a Q(s,a)$ dans chaque case couloir. Pas de flèche tant que la case n'a pas été visitée ou que ses Q sont toutes égales.
- **Courbe par épisode** : nombre de pas (courbe principale, la plus parlante) et récompense totale. Chaque série montre les points bruts en léger, plus une moyenne glissante sur 20 épisodes. Une ligne horizontale en pointillés indique la longueur du plus court chemin S→G (calculée par BFS à la génération), pour qu'on voie la convergence vers l'optimum. Les épisodes tronqués sont marqués d'une autre couleur.
- **Survol d'une case** : affiche les 4 valeurs Q (haut, bas, gauche, droite) de la case.
- Compteurs : numéro d'épisode, pas dans l'épisode courant, ε courant.

## Hors périmètre

- Tout entraînement ou script Python, tout export ONNX.
- Deep Q-Network ou autre approximation par réseau de neurones.
- SARSA, Value Iteration ou comparaison entre algorithmes.
- Éditeur de labyrinthe manuel et labyrinthes prédéfinis.
- Pièges, bonus, environnement stochastique.
- **Mode démo séparé** (« jouer la politique greedy »). La convergence se voit sur les flèches, la heatmap et la courbe.
- Tout mode de comparaison interactif avec le plus court chemin. Le BFS ne sert qu'à tracer la ligne de référence sur la courbe.
- Panneau d'explication théorique intégré au site (la théorie va dans le README).

## Utilisateurs cibles

Les étudiants du cours (Laval), qui lancent le site en local et jouent avec les hyperparamètres pour comprendre l'exploration/exploitation, l'effet de γ et la propagation des valeurs depuis la sortie. Cible : un navigateur desktop.

## Contraintes techniques

- **Stack** : Vite + TypeScript vanilla + Canvas 2D, comme les TP 4 à 7. Aucune dépendance runtime supplémentaire, sauf si c'est justifié (la courbe peut être dessinée à la main sur Canvas, comme `tp4/src/chart.ts`).
- **Organisation du dépôt** (conventions existantes) :
  - dossier `tp8/` avec `index.html`, `vite.config.ts` (`root` = `tp8`, `outDir` = `dist`), `src/`, `README.md` ;
  - dépendances JS dans le `package.json` racine (bun), script `tp8:dev` (et `tp8:build` si besoin) ;
  - tâche `mise run tp8` dans `mise.toml`, avec la description « Démarre le serveur de dev du TP8 (Q-learning dans un labyrinthe) ».
- Découpage suggéré de `src/` : `maze.ts` (génération + dynamique de l'environnement), `agent.ts` (Q-learning), `renderer.ts` (grille, heatmap, flèches), `chart.ts` (courbe), `main.ts` (UI et boucle d'animation), `style.css`.
- Logique RL (environnement + agent) séparée du rendu, pour qu'on puisse la lire comme du code de cours.
- Performance : en mode turbo, un labyrinthe 25×25 doit converger en quelques secondes sans figer l'interface. La boucle est découpée par `requestAnimationFrame`.
- Interface et commentaires en français.

## Livrables

1. Le site `tp8/`, lancé par `mise run tp8`.
2. `tp8/README.md` explicatif :
   - rappel théorique : MDP, état, action, récompense, retour actualisé, fonction Q, équation de Bellman, mise à jour du Q-learning, dilemme exploration/exploitation, ε-greedy et sa décroissance ;
   - lien entre ces notions et ce qu'on voit à l'écran (heatmap = V, flèches = politique, courbe = convergence) ;
   - mode d'emploi du site et expériences suggérées (γ faible vs élevé, ε = 0 sans décroissance, pénalité de mur nulle, etc.).

## Critères de succès

- `mise run tp8` ouvre un site fonctionnel sans étape préalable, en dehors de `mise run prepare`.
- Tout labyrinthe généré a un chemin de S à G.
- Avec les paramètres par défaut, sur un labyrinthe 10×10 en mode turbo, le nombre de pas par épisode descend et se stabilise près de la longueur du plus court chemin. Les flèches forment alors un chemin continu de S à G.
- La heatmap montre un gradient de valeurs qui décroît en s'éloignant de G.
- Play/Pause, pas à pas, vitesse, turbo et reset marchent sans erreur console, et les réglages s'appliquent en direct.
- Le survol d'une case affiche ses 4 valeurs Q, cohérentes avec la flèche affichée.
- Le README permet à un étudiant de comprendre le Q-learning et de relier la théorie à l'interface.

## Points en suspens

Aucun : les derniers choix sont intégrés ci-dessus (défauts et plages des curseurs §4, position de S/G et troncature §1, ligne de référence BFS §6).
