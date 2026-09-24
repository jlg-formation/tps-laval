# TP — Prévision des tickets d’une DSI avec un RNN

## 1. Contexte

Une communauté d'agglomération dispose d'une **Direction des Systèmes d'Information (DSI)** chargée de maintenir de nombreux systèmes et applications métiers.

Les utilisateurs de la collectivité sollicitent régulièrement le support informatique :

* problème de connexion ;
* mot de passe ;
* dysfonctionnement d'une application métier ;
* problème réseau ;
* incident matériel ;
* demande d'assistance ;
* panne d'un service informatique.

Ces sollicitations sont enregistrées dans un outil de gestion de tickets.

La DSI souhaite exploiter l'historique de ces tickets afin **d'anticiper la charge du support**.

L'objectif du TP est de construire un modèle de Deep Learning capable de prévoir le **nombre de tickets qui seront reçus le lendemain**.

---

## 2. Objectif du TP

Nous allons construire une chaîne complète :

```text
Données historiques
       ↓
Préparation des données
       ↓
Séries temporelles
       ↓
RNN avec PyTorch
       ↓
Entraînement
       ↓
Évaluation
       ↓
Export ONNX
       ↓
Application Web
       ↓
Prédiction dans le navigateur
```

Le TP permettra ainsi de parcourir toutes les étapes allant de la donnée brute jusqu'à l'utilisation d'un réseau neuronal dans une application.

---

## 3. Jeu de données

Dans une situation réelle, les données pourraient provenir de l'outil de ticketing utilisé par la collectivité.

Nous pourrions par exemple disposer d'un fichier CSV :

```csv
date,tickets
2026-01-01,42
2026-01-02,38
2026-01-03,35
2026-01-04,47
2026-01-05,61
...
```

Chaque ligne représente :

* une date ;
* le nombre de tickets reçus pendant cette journée.

Si aucun jeu de données réel anonymisé n'est disponible pendant le TP, nous fabriquerons un **jeu de données synthétique mais réaliste**.

Il pourra reproduire certains phénomènes :

* variations selon le jour de la semaine ;
* baisse d'activité pendant les week-ends ;
* périodes de vacances ;
* variations aléatoires ;
* pics correspondant à des incidents importants ;
* évolution progressive de l'activité.

---

## 4. Analyse critique des données

Avant de construire le réseau neuronal, nous examinerons la qualité des données.

Nous chercherons notamment :

* les données manquantes ;
* les valeurs aberrantes ;
* les doublons ;
* les périodes sans données ;
* les changements éventuels de système de ticketing ;
* les événements exceptionnels.

Nous nous demanderons également quelles informations pourraient expliquer les variations observées.

Par exemple :

```text
nombre de tickets
jour de la semaine
jour férié
vacances scolaires
mise en production
maintenance informatique
incident majeur
```

Cette étape permettra de montrer qu'un réseau neuronal ne peut apprendre que ce qui est représenté dans les données qui lui sont fournies.

---

## 5. Transformer les données en séquences

Un RNN travaille naturellement avec des **séquences**.

Nous allons donc transformer notre série :

```text
42, 38, 35, 47, 61, 55, 44, 50, 46...
```

en exemples d'apprentissage.

Avec une fenêtre de 7 jours :

```text
[42, 38, 35, 47, 61, 55, 44] → 50

[38, 35, 47, 61, 55, 44, 50] → 46

[35, 47, 61, 55, 44, 50, 46] → ...
```

Le problème d'apprentissage devient donc :

> **À partir des N derniers jours, prédire le nombre de tickets du jour suivant.**

Mathématiquement :

$$
(x_{t-N+1}, ..., x_{t-1}, x_t)
\longrightarrow
\hat{x}_{t+1}
$$

Nous pourrons expérimenter différentes tailles de fenêtres :

* 7 jours ;
* 14 jours ;
* 30 jours.

---

## 6. Construction du RNN

Le modèle sera développé avec **Python et PyTorch**.

Nous commencerons volontairement avec une architecture très simple :

```text
Séquence des N derniers jours
            ↓
           RNN
            ↓
       État caché
            ↓
     Couche Linear
            ↓
Nombre de tickets prédit
```

Le réseau pourra par exemple contenir :

```python
nn.RNN(
    input_size=1,
    hidden_size=16,
    batch_first=True
)
```

suivi d'une couche :

```python
nn.Linear(16, 1)
```

Le RNN devra apprendre à exploiter l'évolution temporelle du nombre de tickets.

---

## 7. Entraînement

Nous séparerons les données en deux ensembles :

```text
Données historiques

├── Entraînement
│
└── Test
```

Attention : puisqu'il s'agit d'une série temporelle, nous éviterons de mélanger aléatoirement passé et futur.

Le modèle sera entraîné avec une fonction de perte adaptée à un problème de régression, par exemple :

```python
nn.MSELoss()
```

et un optimiseur tel que :

```python
torch.optim.Adam(...)
```

La boucle d'apprentissage permettra de revoir les principales étapes de PyTorch :

```python
prediction = model(X)

loss = criterion(prediction, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 8. Évaluation du modèle

Une fois le réseau entraîné, nous comparerons les valeurs prédites avec les valeurs réellement observées.

Par exemple :

```text
Jour        Réel       Prédit

J+1          41          39
J+2          37          40
J+3          52          49
J+4          46          48
J+5          61          57
```

Nous tracerons également les deux séries sur un graphique :

```text
Nombre
de tickets

   │       réel
60 │        /\       /\
   │   /\  /  \     /  \
50 │  /  \/    \___/
   │
40 │ ----- prédiction -----
   │
   └────────────────────────→ temps
```

L'objectif sera d'analyser les erreurs du réseau plutôt que de regarder uniquement la valeur de la fonction de perte.

---

## 9. Export du modèle en ONNX

Une fois le modèle PyTorch entraîné, nous l'exporterons au format **ONNX**.

La chaîne devient alors :

```text
PyTorch
   ↓
Entraînement
   ↓
Export
   ↓
ticket-rnn.onnx
```

L'intérêt d'ONNX est de pouvoir utiliser le réseau entraîné en dehors de Python.

Nous allons notamment l'exploiter directement depuis une application Web.

---

## 10. Création d'une application Web

Nous réaliserons une petite application avec :

* Vite ;
* TypeScript ;
* ONNX Runtime Web.

L'utilisateur pourra saisir les nombres de tickets observés pendant les derniers jours.

Par exemple :

```text
Tickets des derniers jours

Lundi       [ 41 ]
Mardi       [ 37 ]
Mercredi    [ 45 ]
Jeudi       [ 51 ]
Vendredi    [ 63 ]
Samedi      [ 12 ]
Dimanche    [  8 ]

        [ PRÉDIRE ]

Prévision pour demain :

        44 tickets
```

Le modèle ONNX sera directement chargé par le navigateur.

La chaîne d'inférence sera donc :

```text
Utilisateur
     ↓
Application TypeScript
     ↓
ONNX Runtime Web
     ↓
ticket-rnn.onnx
     ↓
Prédiction
```

Aucun serveur Python ne sera nécessaire pour effectuer la prédiction.

---

## 11. Amélioration du modèle

La première version utilisera volontairement une seule variable :

```text
nombre de tickets
```

Nous réfléchirons ensuite aux informations supplémentaires susceptibles d'améliorer les prédictions.

Par exemple :

```text
nombre de tickets
+
jour de la semaine
+
jour férié
+
vacances
+
mise en production
+
incident majeur
```

Le RNN recevrait alors plusieurs caractéristiques pour chaque instant.

On passerait par exemple de :

```python
input_size=1
```

à :

```python
input_size=5
```

ou davantage.

Cette évolution permettra d'introduire la notion de **série temporelle multivariée**.

---

## 12. Limites et discussion

Le TP devra également permettre de prendre du recul sur le modèle obtenu.

Quelques questions pourront être discutées :

* Combien d'historique faut-il pour entraîner correctement le modèle ?
* Une fenêtre de 7 jours est-elle suffisante ?
* Comment prendre en compte les jours fériés ?
* Comment traiter un incident exceptionnel ?
* Peut-on réellement prévoir un incident informatique qui n'a aucun signe précurseur ?
* Un RNN apporte-t-il réellement quelque chose par rapport à une méthode statistique plus simple ?
* Faudrait-il essayer un GRU ou un LSTM ?
* Comment mesurer si le modèle est suffisamment fiable pour être utile à la DSI ?

---

## 13. Livrables

À la fin du TP, nous disposerons de :

```text
tickets.csv
      ↓
train.py
      ↓
ticket-rnn.pth
      ↓
ticket-rnn.onnx
      ↓
Application Web
      ↓
Prédiction interactive
```

Les stagiaires auront ainsi réalisé une chaîne complète de Deep Learning :

**données → préparation → séquences → RNN → apprentissage → évaluation → ONNX → application Web.**

## Objectif pédagogique final

Au-delà de la prédiction du nombre de tickets, le véritable objectif du TP est de comprendre **pourquoi et comment utiliser un réseau neuronal récurrent lorsque l'ordre temporel des observations contient de l'information**.

Le cas de la DSI fournit un problème concret permettant de relier les concepts du Deep Learning à une problématique directement compréhensible dans le contexte d'une collectivité territoriale.
