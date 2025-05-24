#!/bin/bash
# Script pour envoyer toutes les images d'un dossier à l'API /detect de ia-face-detector

API_URL="http://localhost:5000/detect"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <repertoire_images>"
  exit 1
fi

IMG_DIR="$1"

if [ ! -d "$IMG_DIR" ]; then
  echo "Répertoire $IMG_DIR introuvable."
  exit 1
fi

for img in "$IMG_DIR"/*.{jpg,jpeg,png}; do
  [ -e "$img" ] || continue
  echo "Envoi de $img ..."
  curl -s -X POST "$API_URL" -F "image=@$img" | jq .
done
