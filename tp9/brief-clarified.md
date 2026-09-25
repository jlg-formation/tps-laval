# TP 9 — Apprentissage par renforcement profond : un agent DQN qui joue à Breakout

## Contexte

Le TP 9 du cours (Laval) continue le TP 8. Au TP 8, on faisait du Q-learning tabulaire dans un labyrinthe, avec une case de table Q par état. Ici, l'espace d'états est trop grand pour une table (positions et vitesses continues de la balle, combinaisons de briques). On passe donc au **Deep Reinforcement Learning** : un réseau de neurones approxime la fonction Q.

Comme au TP 8, le TP est un **site web autonome** : l'entraînement se fait entièrement dans le navigateur. Pas de Python, pas d'ONNX.

## Objectif

Montrer en temps réel un agent qui apprend à jouer à **Breakout** (casse-briques) avec un **DQN** (Deep Q-Network), et permettre à l'étudiant :

- de regarder l'agent jouer et progresser (animation normale ou mode turbo) ;
- de suivre l'apprentissage (courbes, Q-valeurs des actions en direct) ;
- de jouer lui-même pour comparer ;
- de sauvegarder et recharger un réseau, ou de charger d'un clic un réseau pré-entraîné livré avec le TP.

## Périmètre fonctionnel

### 1. Environnement : Breakout classique

- Zone de jeu rectangulaire avec murs gauche, droit et haut. Le bas est ouvert : la balle perdue coûte une vie.
- **Mur de briques** : 6 rangées × 10 colonnes, avec une couleur par rangée, comme dans le jeu d'origine.
- **Raquette** horizontale en bas, de largeur fixe. Elle se déplace à vitesse constante et s'arrête contre les bords.
- **Balle** :
  - rebond sur les murs, la raquette et les briques ;
  - l'angle de renvoi dépend du point d'impact sur la raquette (au centre, la balle repart presque verticalement ; sur les bords, elle repart plus à plat), comme dans le jeu d'origine ;
  - une brique touchée disparaît, et la balle rebondit.
- **Vitesse croissante** : la balle accélère par paliers, par exemple après 4 puis 12 renvois, et au premier contact avec les rangées du haut. La vitesse est plafonnée pour que la collision reste fiable (pas d'effet tunnel).
- **Vies** : 3 par partie. Après une vie perdue, la balle repart **automatiquement** depuis la raquette avec un angle tiré au hasard. Il n'y a pas d'action « lancer ».
- **Fin de partie (épisode)** : plus de vies (terminal), mur entièrement vidé (terminal), ou **troncature** au-delà d'un nombre maximal de pas. La troncature évite une balle coincée dans une boucle infinie.
- **Physique à pas de temps fixe**, indépendante du rendu, pour que le mode turbo et l'animation simulent le même jeu. Le seul hasard vient de l'angle de lancement.

### 2. Observation (entrée du réseau)

Un vecteur compact, normalisé autour de [0, 1] ou [−1, 1] :

- position de la balle (x, y) ;
- vitesse de la balle (vx, vy) ;
- position horizontale de la raquette ;
- état des briques : 60 valeurs binaires (1 = brique présente).

Soit **65 entrées**. Pas de pixels, pas de CNN.

### 3. Actions

3 actions discrètes : **gauche**, **rester**, **droite**. L'agent décide à chaque pas de physique.

### 4. Récompenses

| Événement       | Récompense |
| --------------- | ---------- |
| Brique cassée   | +1         |
| Vie perdue      | −1         |
| Autre pas       | 0          |

Pour calculer la cible DQN, **une vie perdue est traitée comme un état terminal** : on coupe le terme $\gamma \max Q(s',\cdot)$. C'est l'astuce classique du DQN sur Atari, qui accélère l'apprentissage. La troncature, elle, garde ce terme, avec la même distinction terminaison / troncature qu'au TP 8.

### 5. Agent DQN

On prend l'algorithme le plus simple qui reste un « vrai » DQN et prolonge directement le Q-learning du TP 8 :

- **Réseau Q** : un MLP 65 → 128 → 128 → 3 avec des ReLU. Il sort une valeur Q par action.
- **Politique ε-greedy**, avec une décroissance **linéaire** de ε en fonction du nombre total de pas.
- **Replay buffer** circulaire (transitions $(s, a, r, s', \text{done})$), dans lequel on tire des mini-batchs uniformes.
- **Réseau cible** : une copie du réseau Q, recopiée tous les C pas.
- **Cible** : $y = r + \gamma \max_{a'} Q_{\text{cible}}(s', a')$ (ou $y = r$ si terminal). **Perte de Huber** entre $Q(s,a)$ et $y$. Optimiseur **Adam**.
- **Warm-up** : pas d'apprentissage tant que le buffer contient moins de N transitions.
- Pas de Double DQN, de Dueling ni de Prioritized Replay (voir « Hors périmètre »).

Valeurs par défaut **indicatives**, à ajuster pendant le développement pour tenir les critères de succès :

| Paramètre                        | Défaut indicatif | Réglable dans l'UI |
| -------------------------------- | ---------------- | ------------------ |
| Taux d'apprentissage (Adam)      | 1e-3             | oui                |
| $\gamma$                         | 0,99             | oui                |
| $\varepsilon$ initial → final    | 1 → 0,05         | oui                |
| Durée de décroissance de ε (pas) | 50 000           | oui                |
| Période de recopie du réseau cible C (pas) | 1 000  | oui                |
| Taille du replay buffer          | 50 000           | non                |
| Taille de mini-batch             | 64               | non                |
| Warm-up (transitions)            | 1 000            | non                |
| Fréquence d'apprentissage        | 1 mise à jour / pas | non             |

Chaque curseur affiche sa valeur et une infobulle d'une ligne. Un bouton **« Valeurs par défaut »** remet les réglages d'origine. Modifier un hyperparamètre l'applique tout de suite, sans reset.

### 6. Modes d'utilisation

- **Entraînement en direct** : l'agent joue et apprend, et la partie s'anime à l'écran.
- **Mode turbo** : l'agent entraîne le plus vite possible, sans animer chaque pas, et la visualisation se rafraîchit de temps en temps.
- **Mode humain** : l'utilisateur joue au **clavier** (flèches gauche/droite) ou à la **souris**. Il joue le même environnement, avec les mêmes règles, et son score s'affiche. Une partie humaine ne nourrit pas le replay buffer.
- **Démo** : l'agent joue en greedy (ε = 0), sans apprendre. Ce mode sert surtout à voir jouer le modèle pré-entraîné ou un modèle rechargé.

### 7. Contrôles

- **Play / Pause** de l'entraînement.
- **Vitesse** : nombre de pas par frame d'animation.
- **Turbo** on/off.
- **Reset** : réinitialise le réseau (poids aléatoires), le buffer, ε et les courbes.
- **Changement de mode** : entraînement, démo, humain.

### 8. Sauvegarde et modèle pré-entraîné

- **Sauvegarder / charger** dans le navigateur (`localStorage`).
- **Exporter / importer** un fichier (téléchargement et upload), au format TensorFlow.js (`model.json` + poids).
- **Modèle pré-entraîné** livré dans `tp9/public/model/`, chargeable d'un clic. Il est produit avec le site lui-même : on entraîne en turbo, puis on exporte. Sa procédure de régénération est décrite dans le README.

### 9. Visualisations

- **Jeu** (Canvas 2D) : briques, raquette, balle, score et vies.
- **Courbe du score par épisode** (briques cassées) : points bruts en léger, plus une moyenne glissante sur 20 épisodes. Une ligne horizontale en pointillés donne le **score moyen d'une politique aléatoire** (mesuré au démarrage sur quelques dizaines de parties), pour qu'on voie tout de suite si l'agent fait mieux que le hasard.
- **Courbe de la perte** (Huber), lissée.
- **Barres des Q-valeurs** des 3 actions dans l'état courant, mises à jour en direct, avec l'action choisie mise en évidence.
- **Compteurs** : épisode, pas total, ε courant, taille du buffer, meilleur score.

## Hors périmètre

- Tout entraînement ou script Python, tout export ONNX.
- Une observation en pixels ou un réseau convolutif.
- Les autres algorithmes (REINFORCE, A2C, PPO…) et toute comparaison entre algorithmes.
- Les variantes de DQN : Double DQN, Dueling, Prioritized Replay, n-step.
- Un mode « humain contre IA » côte à côte.
- L'apprentissage à partir des parties humaines (imitation, démonstrations dans le buffer).
- Les bonus et power-ups, les niveaux multiples, la raquette qui rétrécit, l'éditeur de niveaux.
- La visualisation des activations internes du réseau.
- Un panneau d'explication théorique dans le site (la théorie va dans le README).
- Le mobile et le tactile.

## Utilisateurs cibles

Les étudiants du cours (Laval). Ils lancent le site en local, regardent l'agent apprendre, jouent avec les hyperparamètres et comparent leur propre jeu à celui de l'agent. Cible : un navigateur desktop.

## Contraintes techniques

- **Stack** : Vite + TypeScript vanilla + Canvas 2D, comme les TP 4 à 8.
- **Réseau de neurones** : **TensorFlow.js** (`@tensorflow/tfjs`). C'est la seule dépendance runtime ajoutée, et elle se justifie (autodiff + Adam, sans écrire la rétropropagation à la main). Utiliser `tf.tidy` / `dispose` pour éviter les fuites de mémoire GPU. On garde le backend par défaut (WebGL), et on se rabat sur CPU si WebGL est indisponible.
- **Courbes** dessinées à la main sur Canvas, comme `tp4/src/chart.ts` et `tp8/src/chart.ts`.
- **Organisation du dépôt** (conventions existantes) :
  - dossier `tp9/` avec `index.html`, `vite.config.ts` (`root` = `tp9`, `outDir` = `dist`), `src/`, `public/model/`, `README.md` ;
  - dépendances JS dans le `package.json` racine (bun), scripts `tp9:dev` et `tp9:build` ;
  - tâche `mise run tp9` dans `mise.toml`, avec la description « Démarre le serveur de dev du TP9 (DQN sur Breakout) ».
- Découpage suggéré de `src/` :
  - `breakout.ts` : environnement (physique, règles, observation, récompense), sans aucune dépendance au rendu ;
  - `replay.ts` : replay buffer ;
  - `dqn.ts` : réseau Q, réseau cible, ε-greedy, pas d'apprentissage ;
  - `training.ts` : boucle d'épisodes et baseline aléatoire ;
  - `renderer.ts` : dessin du jeu ;
  - `chart.ts` : courbes et barres de Q-valeurs ;
  - `main.ts` : UI, modes, entrée clavier/souris, boucle d'animation ;
  - `style.css`.
- La logique RL (environnement + agent) reste séparée du rendu, pour qu'on puisse la lire comme du code de cours.
- **Performance** : le mode turbo ne fige pas l'interface. La boucle est découpée en tranches (`requestAnimationFrame` ou `setTimeout`).
- Interface et commentaires en français.

## Livrables

1. Le site `tp9/`, lancé par `mise run tp9`.
2. Le modèle pré-entraîné dans `tp9/public/model/`.
3. `tp9/README.md` explicatif :
   - rappel du TP 8, et pourquoi une table Q ne suffit plus ;
   - approximation de fonction : Q-network ;
   - DQN : replay buffer (décorrélation des transitions), réseau cible (stabilité de la cible), perte de Huber, ε-greedy à décroissance linéaire ;
   - choix de conception de l'environnement : observation vectorielle, 3 actions, récompenses, vie perdue traitée comme terminale ;
   - lien entre la théorie et l'écran (courbe de score contre baseline aléatoire, perte, barres de Q-valeurs) ;
   - mode d'emploi (entraînement, turbo, démo, humain, sauvegarde/chargement) et procédure de régénération du modèle pré-entraîné ;
   - expériences suggérées : γ faible, pas de réseau cible (C = 1), ε constant, taux d'apprentissage trop élevé, etc.

## Critères de succès

- `mise run tp9` ouvre un site fonctionnel sans étape préalable, en dehors de `mise run prepare`.
- Le jeu est jouable par un humain au clavier et à la souris : rebonds cohérents, pas de balle qui traverse une brique ou la raquette, vitesse croissante, 3 vies.
- **Apprentissage** : avec les paramètres par défaut, après **quelques minutes** en mode turbo, la moyenne glissante du score est **nettement au-dessus de la baseline aléatoire** (par exemple au moins 3 fois plus haute).
- Le modèle pré-entraîné, chargé en mode démo, joue nettement mieux que la politique aléatoire.
- Sauvegarde et chargement (`localStorage` et fichier) restituent un agent au même comportement.
- Play/Pause, vitesse, turbo, reset et changement de mode marchent sans erreur console. La mémoire reste stable pendant un long entraînement (pas de fuite de tenseurs).
- Les barres de Q-valeurs se mettent à jour en direct et sont cohérentes avec l'action choisie en mode greedy.
- Le README permet à un étudiant de comprendre le DQN et de relier la théorie à l'interface.

## Points en suspens

- Les valeurs par défaut des hyperparamètres (§5) sont indicatives. Il faut les valider expérimentalement pour tenir le critère « nettement au-dessus du hasard en quelques minutes ».
- Si l'apprentissage est trop lent, une **répétition d'action** (frame skip, par exemple 2 à 4 pas de physique par décision) pourra être ajoutée. Elle devra alors être documentée dans le README.
