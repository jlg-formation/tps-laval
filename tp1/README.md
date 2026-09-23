# TP1 — Le perceptron en images

Un petit laboratoire interactif pour **voir** comment un perceptron (le plus
simple des neurones artificiels) apprend à séparer deux classes de points par
une droite.

## À quoi ça sert ?

Un perceptron prend **2 entrées** (ici les coordonnées `x` et `y` d'un point) et
répond **0 ou 1** (ici : point *blanc* ou point *noir*). Il apprend en ajustant
trois nombres — deux poids `w1`, `w2` et un biais `b` — jusqu'à trouver la
**droite frontière** qui sépare les deux classes.

Cette droite a pour équation :

$$ w_1 \cdot x + w_2 \cdot y + b = 0 $$

Le but du TP est de la voir bouger, itération après itération, jusqu'à séparer
correctement tous les points.

## Démarrer le logiciel

Le projet utilise **Bun** (déjà figé via `mise`). Depuis la **racine du
workspace** (`tps-laval`) :

```bash
# 1. Installer les dépendances (une seule fois)
bun install

# 2. Lancer le serveur de développement
bun run dev
```

Ouvre ensuite l'adresse affichée dans le terminal, en général
<http://localhost:5173/>.

Pour générer une version statique (dossier `dist/`) :

```bash
bun run build
```

## Comment l'utiliser

Au démarrage, un **jeu de démonstration** s'affiche : un amas de points noirs et
un amas de points blancs, déjà séparables par une droite.

### Poser des points

- **Choisir la classe** avec le bouton *Classe active* (bascule *Noir* / *Blanc*).
- **Clic sur une zone vide** : ajoute un point de la classe active.
- **Clic sur un point existant** : le supprime.

### Entraîner

- **Entraîner / Pause** : démarre ou suspend l'apprentissage. La droite rouge se
  redessine à chaque étape : on voit la frontière converger.
- **Pas à pas** : exécute **une seule itération** par clic — idéal pour analyser
  finement chaque mise à jour des poids.

> **Qu'est-ce qu'une itération ?** Ici, une itération = une passe complète sur
> **tous** les points : on accumule les corrections souhaitées, puis on applique
> **une seule** mise à jour de la droite (descente de gradient *batch*).
- **Vitesse** : nombre de corrections appliquées par image (animation plus ou
  moins rapide).
- **Learning rate** : amplitude de chaque correction. Trop grand → ça oscille ;
  trop petit → ça converge lentement. À expérimenter !

### Réinitialiser

- **Reset** : efface tous les points et remet le modèle à zéro.
- **Démo** : recharge le jeu de points de démonstration.

### Lire l'état du modèle

Le panneau de droite affiche en temps réel :

| Champ | Signification |
|-------|---------------|
| `w1`, `w2` | poids des deux entrées |
| `b` | biais |
| Itérations | nombre de corrections effectuées |
| Erreurs | nombre de points actuellement mal classés |

L'apprentissage a réussi quand **Erreurs = 0**.

## Petites expériences à proposer

1. Lance l'entraînement sur la démo et observe `Erreurs` descendre jusqu'à 0.
2. Ajoute un point **du mauvais côté** de la droite et relance : elle se déplace.
3. Rends les deux amas **non séparables linéairement** (mélange-les). Le
   perceptron ne converge jamais : `Erreurs` reste > 0. C'est la limite
   fondamentale d'un perceptron simple.
4. Augmente le *learning rate* au maximum : la droite devient instable.

## Comment ça marche (sous le capot)

À chaque itération, le perceptron parcourt **tous** les points, accumule le
gradient (la somme des corrections souhaitées), puis applique **une seule** mise
à jour des paramètres — c'est la descente de gradient *batch* :

```text
pour chaque point :  erreur = classe_attendue − sortie   (vaut -1, 0 ou +1)
                     g_w1 += erreur × x
                     g_w2 += erreur × y
                     g_b  += erreur
puis, une seule fois (N = nombre de points) :
                     w1 += learning_rate × g_w1 / N
                     w2 += learning_rate × g_w2 / N
                     b  += learning_rate × g_b  / N
```

Si tous les points sont bien classés, le gradient est nul et rien ne change.
Sinon, la droite est poussée dans la bonne direction. Répété assez de fois sur
des données séparables, cet algorithme **converge** vers une frontière correcte.

## Organisation du code

```text
tp1/
├─ index.html          # structure de la page et des contrôles
├─ vite.config.ts      # configuration Vite (root = tp1/)
└─ src/
   ├─ main.ts          # point d'entrée : clics, boucle d'animation
   ├─ perceptron.ts    # le neurone : predict() et règle d'apprentissage
   ├─ geometry.ts      # conversions pixels ↔ [-1, 1] + droite de décision
   ├─ state.ts         # état partagé + points de démonstration
   ├─ renderer.ts      # dessin du canvas (régions, droite, points)
   ├─ ui.ts            # câblage des boutons/sliders et affichages
   └─ style.css        # mise en page
```

Le code est abondamment commenté en français, à visée pédagogique : commence par
[`src/perceptron.ts`](./src/perceptron.ts) pour comprendre l'algorithme.
