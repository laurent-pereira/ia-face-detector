# Utiliser une image Python comme base
FROM python:3.10-slim

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

RUN wget https://github.com/lindevs/yolov8-face/releases/latest/download/yolov8n-face-lindevs.pt -O yolov8n-face.pt

# Copier les fichiers de l'application
COPY . /app

# Installer les dépendances Python
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Commande par défaut pour exécuter le script
CMD ["python", "main.py"]
