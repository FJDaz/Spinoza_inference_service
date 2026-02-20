---
title: Spinoza Inference Service
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
python_version: "3.10"
pinned: false
---

# Spinoza Inference Service

Ce Hugging Face Space héberge un service d'inférence multi-modèles comprenant :
- BERT pour la détection d'intention (vigilance).
- Llama 3B (quantifié 4-bit) pour l'inférence de base.
- Mistral 7B (quantifié 4-bit) pour l'inférence experte.

Le service bascule dynamiquement entre les modèles 3B et 7B en fonction de l'intention détectée.
Il est configuré pour s'exécuter sur Hugging Face Spaces en utilisant Docker, avec le port 7860 exposé pour FastAPI.
Il prend également en charge le déploiement sur RunPod serverless via un interrupteur de variable d'environnement.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
