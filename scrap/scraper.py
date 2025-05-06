import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import time

# URL à scraper
URL = 'https://generated.photos/faces'

# Dossier pour enregistrer les images
IMAGE_DIR = '../images'
os.makedirs(IMAGE_DIR, exist_ok=True)

# Headers pour simuler un vrai navigateur
HEADERS = {'User-Agent': 'Mozilla/5.0'}

def download_image(img_url, filename):
    response = requests.get(img_url, headers=HEADERS)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Image enregistrée : {filename}")
    else:
        print(f"Erreur pour {img_url}")

def scrape_faces(url, max_images=5):
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    img_tags = soup.select('a img')
    count = 0
    
    for img in img_tags:
        img_url = img.get('src')
        if img_url:
            full_url = urljoin(url, img_url)
            filename = os.path.join(IMAGE_DIR, f'face_{count + 1}.jpg')
            download_image(full_url, filename)
            count += 1
            if count >= max_images:
                break
            time.sleep(1)  # Pause pour respecter le serveur

scrape_faces(URL, max_images=5)