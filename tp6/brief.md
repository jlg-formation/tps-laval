# TP — VAE MNIST : explorer visuellement un espace latent

## 1. Objectif

L'objectif de ce TP est de construire un **Variational Autoencoder (VAE)** entraîné sur le dataset **MNIST**, puis de créer une application web permettant d'explorer visuellement son **espace latent**.

Le VAE devra utiliser volontairement un espace latent de seulement **2 dimensions** :

$$
z = (z_1, z_2)
$$

Cette contrainte permettra de représenter directement l'espace latent sous la forme d'un plan 2D.

L'utilisateur pourra déplacer un point sur ce plan. À chaque position du point, le décodeur du VAE générera une image de chiffre manuscrit.

En déplaçant progressivement le point, l'utilisateur devra ainsi pouvoir observer des transformations continues entre différentes formes apprises par le modèle.

Par exemple :

```text
2 → 2 légèrement déformé → forme intermédiaire → 7
```

Le but principal du TP n'est donc pas d'obtenir le meilleur générateur MNIST possible, mais de **comprendre visuellement la notion d'espace latent continu**.

---

# 2. Technologies

## Entraînement

Utiliser :

- Python
- PyTorch
- torchvision
- NumPy
- ONNX pour l'export

Aucune visualisation n'est faite côté Python : tout est affiché dans l'application web.

## Application web

Utiliser :

- Bun
- Vite
- TypeScript
- HTML / CSS
- Canvas 2D
- `onnxruntime-web`

Éviter d'ajouter un framework frontend si celui-ci n'est pas nécessaire.

L'application doit rester simple et pédagogique.

---

# 3. Dataset

Utiliser le dataset **MNIST** fourni par `torchvision`.

Chaque observation contient :

- une image en niveaux de gris ;
- de taille $28 \times 28$ pixels ;
- représentant un chiffre entre 0 et 9 ;
- accompagnée de son label.

Normaliser les pixels dans l'intervalle :

$$
[0,1]
$$

Le téléchargement du dataset doit être automatique lors du premier lancement du script d'entraînement.

---

# 4. Architecture générale

Le système doit suivre cette architecture :

```text
                 ENTRAÎNEMENT

             image MNIST 28×28
                     │
                     ▼
                  Encoder
                     │
              ┌──────┴──────┐
              ▼             ▼
             μ             log σ²
              │             │
              └──────┬──────┘
                     │
             reparameterization
                     │
                     ▼
                z = (z₁,z₂)
                     │
                     ▼
                  Decoder
                     │
                     ▼
             image 28×28
```

L'espace latent doit impérativement avoir :

```python
latent_dim = 2
```

---

# 5. Architecture proposée du réseau

Une architecture MLP simple suffit pour MNIST.

## Encodeur

```text
28 × 28
   │
Flatten
   │
  784
   │
Linear
   │
  256
   │
 ReLU
   │
Linear
   │
   64
   │
 ReLU
   │
 ┌─┴───────────────┐
 │                 │
 ▼                 ▼
μ : 2          logvar : 2
```

L'encodeur doit donc produire deux vecteurs :

$$
\mu = (\mu_1,\mu_2)
$$

et :

$$
\log(\sigma^2) =
(\log(\sigma_1^2),\log(\sigma_2^2))
$$

---

# 6. Reparameterization trick

Pendant l'entraînement, calculer :

$$
z = \mu + \sigma \odot \epsilon
$$

avec :

$$
\epsilon \sim \mathcal{N}(0,I)
$$

et :

$$
\sigma = \exp\left(\frac{1}{2}\log(\sigma^2)\right)
$$

Cette opération permet d'effectuer la rétropropagation malgré l'échantillonnage aléatoire.

Créer une fonction dédiée, par exemple :

```python
def reparameterize(mu, logvar):
    ...
```

---

# 7. Décodeur

Le décodeur reçoit directement :

```text
(z₁, z₂)
```

et reconstruit une image MNIST.

Architecture proposée :

```text
2
│
Linear
│
64
│
ReLU
│
Linear
│
256
│
ReLU
│
Linear
│
784
│
Sigmoid
│
Reshape
│
1 × 28 × 28
```

La sortie doit donc être une image dont les pixels sont compris entre 0 et 1.

---

# 8. Fonction de coût

La loss du VAE doit combiner deux termes.

## Reconstruction

Mesurer la différence entre l'image originale $x$ et l'image reconstruite $\hat{x}$.

Par exemple avec une Binary Cross Entropy :

$$
L_{reconstruction}
=
BCE(x,\hat{x})
$$

## Régularisation KL

Ajouter la divergence de Kullback-Leibler :

$$
L_{KL}
=
-\frac{1}{2}
\sum
\left(
1+\log(\sigma^2)-\mu^2-\sigma^2
\right)
$$

La loss globale devient :

$$
L =
L_{reconstruction}
+
\beta L_{KL}
$$

Commencer avec :

```python
beta = 1.0
```

La valeur pourra éventuellement être modifiée si l'espace latent obtenu est difficile à exploiter.

---

# 9. Entraînement

Créer un script Python :

```text
train.py
```

Il doit :

1. télécharger MNIST si nécessaire ;
2. construire le VAE ;
3. entraîner le modèle ;
4. afficher à chaque époque :
   - la loss totale ;
   - la reconstruction loss ;
   - la KL loss ;
5. sauvegarder les poids PyTorch ;
6. exporter le décodeur en ONNX ;
7. exporter les coordonnées latentes du jeu de test pour l'application web.

Une configuration de départ raisonnable :

```python
batch_size = 128
epochs = 20
learning_rate = 1e-3
latent_dim = 2
```

Utiliser Adam :

```python
torch.optim.Adam(...)
```

Le script doit automatiquement utiliser CUDA si disponible, sinon CPU.

---

# 10. Visualisation de l'espace latent

Après entraînement, faire passer les images du dataset de test dans l'encodeur.

Pour chaque image, récupérer de préférence :

```text
(μ₁, μ₂)
```

plutôt qu'un échantillonnage aléatoire de $z$.

Conserver également le véritable label MNIST.

On obtient alors des données du type :

```text
z1,z2,label
-0.73,1.42,7
-1.12,-0.38,2
0.84,0.17,1
...
```

Exporter ces points dans un fichier consommé par l'application web (par exemple `latent-points.json` dans `public/`).

La visualisation 2D est réalisée **dans l'application web** (Canvas 2D), en arrière-plan du plan latent, pour observer comment le VAE a organisé les chiffres dans son espace latent.

Les différentes classes pourront être distinguées visuellement (une couleur par label).

Attention : ne pas supposer que le VAE va créer dix régions parfaitement séparées. Il s'agit d'un apprentissage non supervisé de la représentation latente.

---

# 11. Export ONNX

Le site web n'a principalement besoin que du **décodeur**.

Exporter celui-ci dans :

```text
decoder.onnx
```

Son interface doit être extrêmement simple.

## Entrée

```text
latent
shape: [batch, 2]
type: float32
```

Par exemple :

```text
[[0.7, -1.2]]
```

## Sortie

Idéalement :

```text
output
shape: [batch, 1, 28, 28]
type: float32
```

Le modèle ONNX doit pouvoir être utilisé indépendamment de Python.

Tester le fichier ONNX depuis Python avec ONNX Runtime avant de passer au développement web.

Vérifier qu'une même coordonnée latente donne des résultats cohérents entre PyTorch et ONNX Runtime.

---

# 12. Application web

Créer ensuite une application web à la racine du dossier du TP (`index.html`, `src/`, `public/`), comme pour les TP précédents,

basée sur :

```text
Bun + Vite + TypeScript
```

Installer :

```text
onnxruntime-web
```

L'application doit charger :

```text
decoder.onnx
latent-points.json
```

directement dans le navigateur, avec `onnxruntime-web`, sans serveur d'inférence.

Fonctionnalités attendues :

- un **plan latent** (Canvas 2D) couvrant les bornes calculées lors de l'entraînement ;
- en fond du plan, au choix :
  - le **nuage** des points du test, coloré par label, avec une légende ;
  - une **grille** de chiffres décodés (par exemple $15 \times 15$), obtenue en un seul appel batché au décodeur ;
- un **point déplaçable** à la souris, au doigt ou aux flèches du clavier ;
- l'**image décodée** $28 \times 28$ correspondant à la position du point, mise à jour en temps réel ;
- l'affichage des **coordonnées** $(z_1, z_2)$ du point.

Une seule inférence doit être en cours à la fois : les déplacements rapides sont regroupés pour que l'image suive le point sans latence cumulée.