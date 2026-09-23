Alors, ce que je voudrais pour le répertoire TP1, c'est faire un petit TP qui montre ce qu'est un perceptron. Je voudrais juste qu'on voie un neurone artificiel avec, on va dire, juste deux entrées qui réponde 0 ou 1, donc qui fasse de la classification.

Je voudrais que tu me traces sur un plan à deux dimensions, on va dire, des points blancs et des points noirs. À la limite, tu peux laisser l'utilisateur cliquer pour ajouter des points ou enlever des points. Une fois que les points sont là, il faudrait qu'on puisse pouvoir entraîner un perceptron de manière à ce que on voie la droite que le perceptron trace devenir la droite de la frontière entre les points blancs et les points noirs. 

---

## Spécifications clarifiées

### Objectif pédagogique
Démontrer visuellement un perceptron à 2 entrées (classification binaire, sortie 0/1) et
comment il apprend une frontière de décision linéaire entre deux classes de points.

### Technologie
- **Vite + TypeScript**, rendu via l'API **Canvas 2D**.
- Le `package.json` est placé **à la racine du workspace** ; Vite est configuré avec
  `root` pointant vers `tp1/` (ou équivalent) pour que le TP vive dans `tp1/`.
- Fichiers principaux : `index.html` + `main.ts` (+ modules TS dédiés).

### Interaction utilisateur
- Un **toggle / bouton** sélectionne la **classe active** (noir ou blanc).
- **Clic gauche sur une zone vide** : ajoute un point de la classe active.
- **Clic gauche sur un point existant** : le supprime.

### Entraînement
- **Animation pas-à-pas** : la droite (frontière) se redessine à **chaque itération**
  de l'algorithme du perceptron, pour visualiser la convergence.

### Contrôles d'interface
- Bouton **Reset** (efface tous les points).
- Bouton **Entraîner / Pause** (démarre/suspend l'animation).
- **Slider de vitesse** d'animation.
- **Learning rate réglable**.

### Affichages temps réel
- Poids **w1, w2** et biais **b** courants.
- Nombre d'**itérations** et nombre d'**erreurs** de classification.

### Qualité du code
- Commentaires de code **détaillés** (visée pédagogique).