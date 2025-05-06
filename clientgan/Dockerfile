# Utilise une image Python légère
FROM python:3.10-slim

# Définit le dossier de travail dans le conteneur
WORKDIR /app

# Copie les fichiers de l’application dans le conteneur
COPY . /app

# Installe Flask
RUN pip install flask

# Expose le port 5000 (Flask par défaut)
EXPOSE 5000

