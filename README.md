# Atelier GAN - Génération de visages

Les tests unitaires sont devenus essentiels dans le développement logiciel moderne pour garantir la qualité, la stabilité et l’évolutivité des projets. Ce tutoriel présente, étape par étape, comment mettre en place et exécuter des tests unitaires sur une API FastAPI à l’aide de `pytest`, en expliquant les concepts clés, les bonnes pratiques et en proposant un exercice pratique pour s’entraîner.


## 1. Explication et utilité des tests unitaires

Les **tests unitaires** sont des scripts permettant de vérifier automatiquement que chaque partie (ou “unité”) du code fonctionne comme prévu.

Ils ont pour principaux objectifs :
- ✅ détecter rapidement les bugs ;
- ✅ garantir que les évolutions futures n’introduisent pas de régressions ;
- ✅ documenter le comportement attendu du code.

---

## 2. Mise en place des tests

1. **Installation des dépendances :**

   Deux approches sont possibles :

   ➤ **Option 1 – Installation manuelle**
   Installer `pytest` :

   ```bash
   pip install pytest
   pip freeze > requirements.txt
   ```

   ➤ **Option 2 – Utilisation du fichier `requirements.txt` présent dans le dépôt GitHub**
   Exécuter la commande suivante :

   ```bash
   pip install -r requirements.txt
   ```

   Cette commande installera toutes les dépendances listées, y compris `pytest`.

2. **Organisation des fichiers :**

   Le fichier `test_api.py` doit être placé dans le répertoire /app, dans un sous-dossier `tests/`.
   Il est nécessaire de s’assurer que le fichier `main.py` (contenant l’API) soit accessible.

   Exemple d’arborescence :

   ```
   /app
   ├── main.py
   ├── test_api.py
   ```

---

## 3. Explication sur les `assert`

Les **assertions** (`assert`) sont des vérifications qui permettent de confirmer qu’une condition est vraie.
En cas d’échec, le test est signalé comme échoué.

Exemples présents dans le fichier `test_api.py` :

* `assert response.status_code == 200` → vérifie que le serveur répond avec un code 200 (OK).
* `assert "API Génération" in response.text` → vérifie que le texte attendu est bien présent dans la réponse.
* `assert len(response.content) > 1000` → vérifie que l’image générée n’est pas vide.

---

## 4. Code complet pour le test `test_generate_image()`

Voici le code complet permettant de tester la génération d’image (à inclure dans le fichier `test_api.py`) :

```python
import sys
import os
from fastapi.testclient import TestClient
from main import app

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
client = TestClient(app)

def test_generate_image():
    response = client.get("/generate_image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert len(response.content) > 1000  # Vérifie que l'image n'est pas vide
```

Ce test :

* appelle la route `/generate_image`,
* vérifie que la réponse est correcte et contient une image JPEG,
* s’assure que l’image n’est pas vide.

---

## 5. Exécution du code et vérification des résultats

1. Depuis un terminal, se placer à la racine du projet :

   ```bash
   cd /<chemin_projet>/app
   ```

2. Exécuter les tests avec pytest :

   ```bash
   pytest
   ```

3. Vérifier les résultats affichés :
   Un exemple de sortie attendue :

   ```
   ================= test session starts ==================
   collected 2 items

   test_api.py ..                                      [100%]

   ================== 2 passed in X.XXs ==================
   ```

   Cela indique que les deux tests (`test_index` et `test_generate_image`) ont réussi.

---

## 6. Exercice : ajouter et exécuter le test `test_index()`

Pour compléter l’exercice :

1. Ajouter le test `test_index()` dans le fichier `test_api.py`.
2. Vérifier que ce test :

   * appelle la route `/`,
   * vérifie que le code retourné est 200,
   * vérifie que le texte `"API Génération"` est présent dans la réponse grâce à l'attribut `response.text`.
3. Exécuter à nouveau `pytest` et s’assurer que tous les tests passent.

---

## Pour aller plus loin

Pour approfondir l’utilisation de `pytest` et découvrir l’ensemble de ses fonctionnalités (paramétrage avancé, fixtures, tests paramétrés, intégration continue, plugins, etc.), il est recommandé de consulter le site officiel : [https://pytest.org](https://pytest.org). Ce site propose une documentation complète, des exemples pratiques, ainsi que des guides pour améliorer et structurer les tests au sein des projets Python. La documentation détaillée est également accessible directement à l’adresse : [https://docs.pytest.org](https://docs.pytest.org).
