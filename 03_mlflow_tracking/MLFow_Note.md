# MLFow

# 🧠 Note : Bonnes pratiques pour utiliser MLflow en Machine Learning

---

## 🎯 Objectif

MLflow est un outil essentiel pour **suivre, comparer, déployer et reproduire** les expériences de machine learning. Cette note propose les **bonnes pratiques** pour structurer un projet, enregistrer proprement ses runs, et tirer le meilleur parti de l’outil.

---

## ✅ 1. 📁 **Organisation du projet**

### Structure de base recommandée :

```
bash
CopierModifier
ml_project/
│
├── train.py                     # Script principal
├── requirements.txt             # Dépendances
├── mlruns/                      # Stockage des runs (généré)
├── models/                      # Export de modèles (optionnel)
├── data/                        # Datasets (optionnel)
├── plots/                       # Graphiques loggés
└── notebook_exploration.ipynb   # Pour visualisation interactive

```

- 💡 Regroupe les runs dans un dossier `mlruns/`
- 💡 Utilise `git` pour versionner le code source

---

## ✅ 2. 🧪 **Utiliser `mlflow.start_run()` correctement**

Toujours encapsuler chaque expérience :

```python
python
CopierModifier
with mlflow.start_run(run_name="nom_du_run"):
    ...

```

- `run_name` permet d’identifier facilement dans l’UI
- Tu peux aussi ajouter des **tags** pour filtrer (ex: `user`, `stage`, `dataset_version`)

---

## ✅ 3. 📊 **Logger tout ce qui compte**

| Ce que tu dois logger | Exemple |
| --- | --- |
| Paramètres | `mlflow.log_param("C", 0.1)` |
| Métriques | `mlflow.log_metric("accuracy", acc)` |
| Artéfacts | `mlflow.log_artifact("confusion_matrix.png")` |
| Modèles | `mlflow.sklearn.log_model(model, "model")` |

### 💡 Bonus : logger signature et exemple

```python
python
CopierModifier
from mlflow.models.signature import infer_signature
signature = infer_signature(X_test, y_pred)
mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_test.iloc[:1])

```

---

## ✅ 4. 🔁 **Reproductibilité**

- Utilise `random_state` partout : modèles, split, etc.
- Stocke la version du dataset (via param ou tag)
- Écris les versions des dépendances dans `requirements.txt`

---

## ✅ 5. 🚀 **Comparaison d’expériences**

Depuis l’interface MLflow :

- Compare plusieurs runs sur des **métriques**
- Trie ou filtre par tags/paramètres
- Visualise les artéfacts (images, fichiers CSV…)

---

## ✅ 6. 🧪 **Exemples d'artéfacts utiles à logger**

- Courbes d’apprentissage (loss, accuracy)
- Matrices de confusion
- Visualisation des embeddings
- Rapport HTML (classification_report de sklearn)
- Modèles `.pkl` ou ONNX

---

## ✅ 7. 🧰 **Déploiement & tracking avancé**

- Active le **model registry** pour valider/stager/promouvoir des modèles
- Utilise `mlflow.set_tracking_uri()` pour enregistrer sur un serveur distant ou base SQLite
- Servir un modèle avec :

```bash
bash
CopierModifier
mlflow models serve -m runs:/<run_id>/model --port 1234

```

---

## ✅ 8. ⚠️ Éviter les pièges classiques

| Mauvaise pratique | Pourquoi c'est problématique |
| --- | --- |
| Ne pas logger les paramètres | Impossible de reproduire |
| Ne pas fixer le random seed | Résultats non stables |
| Logger manuellement des fichiers inutiles | UI surchargée |
| Lancer `mlflow ui` dans un mauvais dossier | Aucun run visible |

---

## 🔚 Conclusion

MLflow est un **compagnon indispensable** pour tout projet machine learning sérieux. Utilisé correctement, il te garantit :

- 🧭 Suivi clair
- 📈 Comparaison rigoureuse
- 🔁 Reproductibilité
- 🔒 Traçabilité des modèles déployés

## Cas concret

## 🎯 Objectif

Utiliser un **modèle de classification (Logistic Regression)** sur le dataset `Iris` pour illustrer pas à pas **les meilleures pratiques MLflow**, **jusqu'au déploiement**.

---

## ✅ 1. 📁 Organisation du projet

**Structure recommandée :**

```
vbnet
CopierModifier
mlflow_demo/
│
├── train.py                      ← Script d'entraînement
├── requirements.txt              ← Dépendances
├── plots/                        ← Graphiques
└── mlruns/                       ← Run tracking (auto-généré)

```

👉 **Exemple concret :**

Aucun besoin de fichier lourd. Le modèle utilisé :

```python
python
CopierModifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

```

---

## ✅ 2. 🧪 Utiliser `mlflow.start_run()`

> Bonne pratique : chaque expérience doit être encapsulée
> 

```python
python
CopierModifier
with mlflow.start_run(run_name="logreg_iris"):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

```

---

## ✅ 3. 📊 Logger les paramètres, métriques, artéfacts et modèles

### Exemple :

```python
python
CopierModifier
from sklearn.metrics import accuracy_score
import mlflow.sklearn

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

mlflow.log_param("max_iter", 200)
mlflow.log_metric("accuracy", acc)
mlflow.sklearn.log_model(model, "model")

```

### Logger un graphe :

```python
python
CopierModifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
plt.savefig("plots/cm.png")
mlflow.log_artifact("plots/cm.png")

```

---

## ✅ 4. 🔁 Reproductibilité

> Bonne pratique : fixer les aléas pour des résultats stables
> 

```python
python
CopierModifier
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
model = LogisticRegression(max_iter=200, random_state=42)

```

Et dans `requirements.txt` :

```
txt
CopierModifier
mlflow
scikit-learn==1.4.2
matplotlib
seaborn

```

---

## ✅ 5. 🚀 Comparaison de modèles

On compare ici deux classifieurs :

```python
python
CopierModifier
models = {
    "logreg": LogisticRegression(max_iter=200, random_state=42),
    "svc": SVC(kernel="linear", probability=True, random_state=42)
}

for name, clf in models.items():
    with mlflow.start_run(run_name=name):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

```

🧭 **Compare dans l’interface** MLflow UI :

👉 [http://localhost:5000](http://localhost:5000/)

---

## ✅ 6. 🧰 Logger la signature + exemple pour servir plus tard

```python
python
CopierModifier
from mlflow.models.signature import infer_signature
signature = infer_signature(X_test, y_pred)

mlflow.sklearn.log_model(
    model,
    "model",
    signature=signature,
    input_example=X_test[:1]
)

```

Cela te permet de :

- Recharger automatiquement le modèle dans `mlflow.pyfunc.load_model()`
- Le servir plus tard via API avec typage clair

---

## ✅ 7. 📦 Déploiement local simple (serveur REST)

### Étape 1 : récupérer l’`run_id` (dans l’interface MLflow)

### Étape 2 : servir ton modèle

```bash
bash
CopierModifier
mlflow models serve -m runs:/<RUN_ID>/model --port 1234

```

Par exemple :

```bash
bash
CopierModifier
mlflow models serve -m runs:/9e9f3c7b1e36466b93d1a0c82bdb1be1/model --port 1234

```

Puis tu peux faire un POST :

```bash
bash
CopierModifier
curl -X POST http://127.0.0.1:1234/invocations \
  -H "Content-Type: application/json" \
  -d '{"data": [[5.1, 3.5, 1.4, 0.2]]}'

```

---

### 🔚 Résultat

- ✅ Projet bien structuré
- ✅ Run bien traqué avec paramètres + métriques
- ✅ Comparaison possible
- ✅ Prêt pour déploiement REST

## ✅ Contenu complet du `train.py` (modèle Iris + MLflow)

```python
python
CopierModifier
# train.py

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from mlflow.models.signature import infer_signature
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Chargement du dataset
X, y = load_iris(return_X_y=True)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Crée le dossier pour les artéfacts (graphes)
os.makedirs("plots", exist_ok=True)

# Début du tracking MLflow
with mlflow.start_run(run_name="logreg_iris"):

    # 🔁 Reproductibilité et initialisation
    model = LogisticRegression(max_iter=200, random_state=42)

    # Entraînement
    model.fit(X_train, y_train)

    # Prédiction
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # 🧾 Logging
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 200)
    mlflow.log_metric("accuracy", acc)

    # Signature pour déploiement
    signature = infer_signature(X_test, y_pred)

    # Log du modèle
    mlflow.sklearn.log_model(
        model, "model", signature=signature, input_example=X_test[:1]
    )

    # 🎨 Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()

    # Sauvegarde + log
    fig_path = "plots/confusion_matrix.png"
    plt.savefig(fig_path)
    mlflow.log_artifact(fig_path)

print("✅ Entraînement terminé. Visualise sur http://localhost:5000")

```

---

## 📌 Ce que ce `train.py` couvre :

| Étape | Fait ? |
| --- | --- |
| Chargement des données Iris | ✅ |
| Split train/test | ✅ |
| Reproductibilité | ✅ |
| Entraînement d’un modèle | ✅ |
| Logging paramètre + métrique | ✅ |
| Logging du modèle | ✅ |
| Signature + input_example | ✅ |
| Graphique (matrice confusion) | ✅ |
| Sauvegarde + artifact dans MLflow | ✅ |