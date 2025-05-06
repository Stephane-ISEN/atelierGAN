from flask import Flask, render_template_string

app = Flask(__name__)

# URL de ton API GAN FastAPI
GAN_API_URL = 'http://localhost:8081/generate_image'

# Template HTML avec JavaScript
TEMPLATE = '''
<!doctype html>
<html>
<head>
    <title>Test API GAN</title>
    <style>
        body { text-align: center; margin-top: 50px; }
        img { cursor: pointer; max-width: 500px; }
    </style>
</head>
<body>
    <h1>Clique sur l’image pour en générer une nouvelle</h1>
    <img id="gan-image" src="{{ image_url }}" onclick="refreshImage()">
    <script>
        function refreshImage() {
            const img = document.getElementById('gan-image');
            img.src = '{{ image_url }}?t=' + new Date().getTime(); // ajoute timestamp pour forcer le rafraîchissement
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(TEMPLATE, image_url=GAN_API_URL)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
