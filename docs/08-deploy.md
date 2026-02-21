# Guide de Déploiement sur Hugging Face Space (Inference Container)

## Étapes de préparation (Build & Test Local)

1.  **Vérification de l'image Docker** :
    Le `Dockerfile` utilise le secret `HF_TOKEN` pendant le build pour télécharger les modèles.

2.  **Configuration locale** :
    - Renommez `.env.example` en `.env`.
    - Ajoutez votre `HF_TOKEN` dans le fichier `.env`.

3.  **Lancement local** :
    ```bash
    docker compose up --build
    ```

## Déploiement Humain sur Hugging Face Spaces

### 1. Synchronisation du Code (Déclencheur de Build)
Hugging Face Spaces re-construit votre conteneur à chaque fois que vous poussez sur sa branche `main`.
Si vous avez configuré le Space comme un remote git (souvent nommé `hf`), utilisez :
```bash
git push hf main
```
*Si vous utilisez une synchro automatique depuis GitHub, assurez-vous que votre push GitHub est bien passé.*

### 2. Configuration des Secrets (CRITIQUE)
**C'est l'étape essentielle pour que le build fonctionne :**

- Allez dans les **Settings** de votre Space sur Hugging Face.
- Dans la section **Variables and secrets**, cliquez sur **New Secret**.
- **Key** : `HF_TOKEN`
- **Value** : (Votre jeton d'accès Hugging Face avec les permissions 'read' ou 'write')

*Note : HF Spaces expose automatiquement les Secrets nommés `HF_TOKEN` dans `/run/secrets/HF_TOKEN` pendant le build Docker.*

### 2. Configuration des Variables d'Environnement
Toujours dans **Settings**, ajoutez les variables suivantes (en tant que Variables, pas Secrets) :
- `HYBRID_REMOTE_GENERATION` : `true`
- `REMOTE_ENGINE_URL` : (L'URL de votre moteur distant)

### 3. Explication Technique
Le `Dockerfile` est configuré pour lire le secret pendant la construction via :
```dockerfile
RUN --mount=type=secret,id=HF_TOKEN \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN) && \
    huggingface-cli login --token $HF_TOKEN && \
    python download_models.py
```
Cela permet de s'authentifier sans laisser le token dans les couches de l'image Docker finale.

## Résolution des problèmes fréquents
- **Erreur "No such file or directory" pour HF_TOKEN** : C'est signe que le **Secret** n'a pas été défini dans les paramètres du Space. Les variables classiques ne sont pas accessibles pendant `docker build`.
- **Modèle non trouvé** : Vérifiez que le `BASE_MODEL` dans `app/inference_server.py` est correct et accessible avec votre token.
