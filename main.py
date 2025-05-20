import cv2
import os
import boto3
import logging
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Démarrage de l'application...")

# Initialisation du client MinIO
MINIO_ENDPOINT = os.environ.get('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS_KEY = os.environ.get('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.environ.get('MINIO_SECRET_KEY', 'minioadmin')
MINIO_BUCKET = os.environ.get('MINIO_BUCKET', 'faces')
MINIO_BUCKET_MEDIA = os.environ.get('MINIO_BUCKET_MEDIA', 'media')

s3 = boto3.client(
    's3',
    endpoint_url=f'http://{MINIO_ENDPOINT}',
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY
)

# Création des buckets si nécessaire
for bucket in [MINIO_BUCKET, MINIO_BUCKET_MEDIA]:
    try:
        s3.head_bucket(Bucket=bucket)
    except:
        s3.create_bucket(Bucket=bucket)

# Chargement du modèle YOLOv8n
model = YOLO('yolov8n-face.pt')  # Modèle spécifique pour la détection de visages

def process_image_file(local_image_path, media_id):
    """Traite une image locale pour détecter les visages et les sauvegarder sur MinIO."""
    image = cv2.imread(local_image_path)
    results = model(image)[0]
    faces_saved = []

    for i, box in enumerate(results.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        face = image[y1:y2, x1:x2]
        face_filename = f"{media_id}_{i}.jpg"
        cv2.imwrite(face_filename, face)
        try:
            s3.upload_file(face_filename, MINIO_BUCKET, face_filename)
            logger.info(f"Visage {face_filename} envoyé avec succès sur MinIO.")
            faces_saved.append(face_filename)
        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de {face_filename} sur MinIO : {e}")
        finally:
            if os.path.exists(face_filename):
                os.remove(face_filename)
    logger.info(f"{len(results.boxes)} visages détectés et envoyés vers MinIO.")
    return faces_saved

# --- API Flask ---
app = Flask(__name__)

@app.route("/detect", methods=["POST"])
def detect_faces():
    """
    Reçoit une image en POST (multipart/form-data, champ 'image') et lance le traitement.
    Stocke l'image originale dans le bucket 'media' sur MinIO.
    Retourne la liste des fichiers de visages stockés sur MinIO.
    """
    if 'image' not in request.files:
        return jsonify({"error": "Aucun fichier image fourni"}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Nom de fichier vide"}), 400

    filename = secure_filename(file.filename)
    local_path = f"/tmp/{filename}"
    file.save(local_path)
    media_id = os.path.splitext(filename)[0]

    # Stockage de l'image originale dans le bucket 'media'
    try:
        s3.upload_file(local_path, MINIO_BUCKET_MEDIA, filename)
        logger.info(f"Image originale {filename} envoyée avec succès sur MinIO (bucket media).")
    except Exception as e:
        logger.error(f"Erreur lors de l'envoi de l'image originale {filename} sur MinIO : {e}")

    faces = process_image_file(local_path, media_id)

    if os.path.exists(local_path):
        os.remove(local_path)

    return jsonify({"faces": faces})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
