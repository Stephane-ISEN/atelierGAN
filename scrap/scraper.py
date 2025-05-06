import requests

url = 'https://generated.photos/faces'
response = requests.get(url)

if response.status_code == 200:
    print("Succès ! Contenu de la page récupéré.")
    print(response.text[:500])  # Affiche les 500 premiers caractères
else:
    print(f"Erreur : statut {response.status_code}")
