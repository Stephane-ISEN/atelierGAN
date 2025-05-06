# Atelier GAN - Les tests automatiques

Les tests automatiques permettent de vérifier en continu que le code fonctionne comme attendu, sans intervention manuelle.
Ils sont essentiels pour détecter rapidement des régressions, valider des évolutions et garantir la stabilité d’un projet, en particulier lorsqu’il implique plusieurs contributeurs.

## Qu’est-ce que Git ?

Git est un système de gestion de versions.
Il permet de **suivre l’historique des fichiers**, **travailler à plusieurs** sur un même projet et **revenir en arrière** en cas d’erreur.

---

## Comment installer Git

✅ **Sur Linux (Debian/Ubuntu) :**

```bash
sudo apt update
sudo apt install git
```

✅ **Sur Windows :**

1. Va sur [https://git-scm.com](https://git-scm.com)
2. Clique sur **Download** (le site détectera ton OS)
3. Lance l’installeur et laisse les options par défaut (ça suffit pour commencer)

---

## Initialiser un dépôt Git

Dans le dossier de ton projet :

```bash
git init
```

Cela crée un dossier `.git` qui stocke l’historique et les métadonnées.

---

## Écrire un `.gitignore`

Crée un fichier nommé `.gitignore` dans ton projet.
Dedans, liste les fichiers/dossiers à ignorer, par exemple :

```
.venv/
.pytest_cache/
__pycache__/
app/stylegan3-r-ffhqu-256x256.pkl
```

---

## Créer une branche

```bash
git checkout -b nouvelle-branche
```

Exemple :

```bash
git checkout -b feature-login
```

---

## Ajouter et commiter le code

1. **Ajouter les fichiers modifiés au suivi :**

```bash
git add .
```

2. **Créer un commit avec un message :**

```bash
git commit -m "Ajout de la fonctionnalité login"
```

---

## Qu’est-ce que GitHub ?

GitHub est une plateforme en ligne pour héberger tes dépôts Git.
Il permet de **partager ton code**, **collaborer à plusieurs** et **gérer les issues/pull requests**.

---

## Relier un dépôt distant sur GitHub

1. Crée un dépôt sur GitHub (via le site) sans cocher “Initialize with README”.

2. Dans ton terminal :

```bash
git remote add origin https://github.com/ton-utilisateur/ton-repo.git
```

---

## Envoyer le code

Pour **pousser (push) une branche vers un remote** (par exemple sur GitHub), voici la commande :

```bash
git push -u origin nom-de-ta-branche
```

✅ Exemple :
```bash
git push -u origin feature-login
```

- `origin` → c’est le nom par défaut du dépôt distant.
- `nom-de-ta-branche` → remplace par le nom réel de ta branche.
- `-u` → ça crée un lien entre ta branche locale et la branche distante (pratique, car après tu pourras juste faire `git push` sans préciser le reste).

**Après le premier push :**
Tu pourras simplement faire :

```bash
git push
```

Si la branche existe déjà sur le remote, tu peux aussi utiliser :

```bash
git push origin nom-de-ta-branche
```
---

## Explication sur les tests automatiques avec GitHub Actions

GitHub Actions est une plateforme d’intégration et de déploiement continu (CI/CD) intégrée à GitHub.
Elle permet d’exécuter automatiquement des tâches (comme des tests) à chaque événement déclenché sur le dépôt (par exemple un push ou une pull request).
Ainsi, dès qu’un collaborateur pousse du code, les tests sont exécutés sur des serveurs GitHub, garantissant que les changements n’introduisent pas d’erreurs.

---

## Comment mettre en place le GitHub Action dans le projet

Pour mettre en place les tests automatiques avec GitHub Actions :

1. Créer un fichier YAML de configuration dans le répertoire suivant :

```
.github/workflows/tests.yaml
```

2. Définir dans ce fichier le workflow, c’est-à-dire :

   * les événements qui déclenchent le workflow (par exemple les pushs sur certaines branches) ;
   * les environnements d’exécution (par exemple Ubuntu, Python 3.12) ;
   * les étapes à suivre (par exemple installer les dépendances, exécuter les tests).

3. Commiter et pousser ce fichier dans le dépôt GitHub.

---

## Code et explication du fichier YAML

Voici le contenu du fichier `tests.yaml` avec explication :

```yaml
name: Run tests API-GAN

on:
  push:
    branches:
      - '**'  # Déclenche le workflow sur toutes les branches

jobs:
  test:
    runs-on: ubuntu-latest  # Utilise une machine virtuelle Ubuntu pour exécuter le job

    steps:
      - name: Checkout repository
        uses: actions/checkout@v3  # Clone le dépôt sur la machine GitHub

      - name: Set up Python
        uses: actions/setup-python@v4  # Installe Python sur la machine
        with:
          python-version: '3.12'  # Version spécifique de Python

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt  # Installe les dépendances du projet

      - name: Run tests
        run: |
          cd app
          pytest  # Exécute les tests avec pytest dans le dossier app
```

---

## Que se passe-t-il au push et comment le vérifier

À chaque push sur une branche :

1. GitHub déclenche automatiquement le workflow défini dans `tests.yaml`.
2. Les étapes sont exécutées dans l’ordre : clone du dépôt, installation de Python, installation des dépendances, puis exécution des tests.
3. GitHub affiche le statut du workflow (succès ou échec) directement dans l’onglet **Actions** du dépôt.
4. Dans l’onglet **Actions**, il est possible de cliquer sur un run spécifique pour :

   * consulter les étapes exécutées ;
   * lire les logs détaillés ;
   * identifier les éventuelles erreurs.

En cas d’échec, le workflow apparaît en rouge, et GitHub peut même afficher un résumé directement dans les pull requests, permettant aux équipes de réagir avant de fusionner le code.

---

## Pour aller plus loin

Pour approfondir vos connaissances sur GitHub Actions et exploiter pleinement ses fonctionnalités, il est recommandé de consulter la documentation officielle. Celle-ci offre une vue d'ensemble complète, des guides pratiques, des exemples de workflows, ainsi que des informations détaillées sur la syntaxe YAML et les événements déclencheurs.

* **Documentation officielle GitHub Actions** : [https://docs.github.com/actions](https://docs.github.com/actions)
