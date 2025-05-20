# ia-face-detector

Ce projet fournit un service de détection de visages via une API HTTP, utilisant Docker Compose pour intégrer MinIO (stockage objet) et un conteneur de détection de visages personnalisé.

## Prérequis
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Installation & Lancement

1. **Cloner le dépôt** (si ce n'est pas déjà fait) :
   ```bash
   git clone <your-repo-url>
   cd ia-face-detector
   ```

2. **Construire et démarrer les services :**
   ```bash
   docker-compose up --build
   ```
   Cela démarre deux services :
   - `minio` : stockage objet accessible sur http://localhost:9000 (console : http://localhost:9001)
   - `ia-face-detector` : service de détection de visages

3. **Accéder à la console MinIO :**
   - URL : [http://localhost:9001](http://localhost:9001)
   - Utilisateur : `minioadmin`
   - Mot de passe : `minioadmin`

## Variables d'environnement
Le service `ia-face-detector` utilise les variables suivantes (voir `docker-compose.yaml`) :
- `MINIO_ENDPOINT` : minio:9000
- `MINIO_ACCESS_KEY` : minioadmin
- `MINIO_SECRET_KEY` : minioadmin
- `MINIO_BUCKET` : faces
- `MINIO_BUCKET_MEDIA` : media

## Exemple d'utilisation

1. **Envoyer une image à l'API**
   - Faites une requête HTTP POST sur l'endpoint du service (par défaut : `http://localhost:5000/detect`) avec une image en pièce jointe (multipart/form-data).
   - Exemple avec `curl` :
     ```bash
     curl -X POST -F "image=@example.jpg" http://localhost:5000/detect
     ```

2. **Traitement et stockage**
   - Le service traite l'image reçue, effectue la détection de visages, puis stocke le résultat dans le bucket `faces` sur MinIO.

3. **Récupérer le résultat**
   - Connectez-vous à la console MinIO ou utilisez le client `mc` pour télécharger le résultat depuis le bucket `faces`.

## Arrêter les services
Pour arrêter et supprimer les conteneurs :
```bash
docker-compose down
```

## Remarques
- Assurez-vous que les ports 5000, 9000 et 9001 sont libres sur votre machine.
- Vous pouvez modifier `main.py` pour personnaliser la logique de détection.

---

Pour plus de détails, consultez les fichiers `docker-compose.yaml` et `main.py`.
