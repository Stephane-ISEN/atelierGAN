import torch
import legacy
from PIL import Image
import io
from fastapi import FastAPI, Response
import os
from datetime import datetime

# Crée un répertoire 'images' pour stocker les images générées si absent
IMAGE_DIR = "../images"

# Initialise l’application FastAPI
app = FastAPI()

# Charge le modèle StyleGAN au démarrage, sur CPU
def load_generator(pkl_path="./stylegan3-r-ffhqu-256x256.pkl"):
    device = torch.device('cpu')
    with open(pkl_path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    return G

# Fonction pour générer et sauvegarder une image, puis retourner ses octets
def generate_stylegan_image(G) -> bytes:
    # Crée un vecteur bruit aléatoire
    z = torch.randn([1, G.z_dim], device='cpu')
    c = None  # pas de conditionnement
    # Génère une image avec le modèle
    img = G(z, c, noise_mode='const')[0]
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.permute(1, 2, 0).cpu().numpy()
    pil_img = Image.fromarray(img, 'RGB')

    # Crée un nom de fichier unique basé sur l’horodatage
    filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)
    # Sauvegarde l’image localement
    pil_img.save(filepath, format='JPEG')

    # Convertit l’image en bytes pour la réponse API
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG')
    return buffer.getvalue()

# Charge le modèle une fois au démarrage
gan = load_generator()

@app.get("/")
async def index():
    return "API Génération d'image avec sauvegarde locale !"

@app.get("/generate_image")
async def generate_image():
    """
    Génère une image GAN, la sauvegarde dans le dossier /images,
    et renvoie l’image au format JPEG.
    """
    image_data = generate_stylegan_image(gan)
    return Response(content=image_data, media_type="image/jpeg")
