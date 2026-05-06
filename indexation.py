import pandas as pd
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# --- 1. LES OUTILS ---
def chunker(texte, taille_max=500, overlap=50):
    """Découpe un texte en morceaux avec chevauchement."""
    chunks = []
    debut = 0
    if len(texte) <= taille_max:
        return [texte]
    
    while debut < len(texte):
        fin = debut + taille_max
        chunk = texte[debut:fin]
        chunks.append(chunk)
        debut += (taille_max - overlap)
    return chunks

# --- 2. CHARGEMENT ---
df = pd.read_csv("data/tmdb_5000_movies.csv")
# On enlève les films sans résumé
df = df.dropna(subset=['overview'])

print(f"Nombre de films à traiter : {len(df)}")

# --- 3. TRANSFORMATION & CHUNKING ---
documents = []

for index, row in df.iterrows():
    # Nettoyage des genres
    try:
        data_genres = json.loads(row['genres'])
        genres_propres = ", ".join([g['name'] for g in data_genres])
    except:
        genres_propres = "Inconnu"

    # On prépare le texte complet du film
    texte_film = f"Titre: {row['title']}. Genre: {genres_propres}. Résumé: {row['overview']}"
    
    # On découpe ce texte en morceaux (chunks)
    morceaux = chunker(texte_film, taille_max=500, overlap=50)

    # Pour chaque morceau, on crée un document structuré
    for i, morceau in enumerate(morceaux):
        doc = {
            "id": f"doc_{index}_chunk_{i}",
            "contenu": morceau,
            "metadata": {
                "source": "tmdb_5000_movies.csv",
                "titre": row['title'],
                "note": row['vote_average'],
                "langue": row['original_language'],
                "genres": genres_propres
            }
        }
        documents.append(doc)

print(f"✅ Terminé ! {len(documents)} morceaux de texte prêts pour l'IA.")

# --- ÉTAPE 3 : CRÉATION DES EMBEDDINGS ---

# 1. On charge le modèle (le cerveau)
modele = SentenceTransformer("all-mpnet-base-v2")

# 2. On définit la fonction de transformation
def embedder_chunks(chunks: list[str], modele) -> np.ndarray:
    print(f"\nEncodage de {len(chunks)} morceaux en cours...")
    # C'est cette ligne qui crée les 768 dimensions pour chaque chunk
    vecteurs = modele.encode(chunks, show_progress_bar=True)
    return vecteurs

# 3. On prépare la liste des textes à envoyer au modèle
textes_a_traiter = [doc['contenu'] for doc in documents]

# 4. On lance la transformation
matrice_finale = embedder_chunks(textes_a_traiter, modele)

# 5. On affiche le résultat final pour vérifier
print(f"✅ Terminé ! Forme de la matrice : {matrice_finale.shape}")

# --- ÉTAPE 4 : CRÉATION ET PERSISTANCE DE LA BASE FAISS ---

def creer_index_faiss(vecteurs: np.ndarray) -> faiss.Index:
    """Crée un index FAISS à partir des vecteurs."""
    dimension = vecteurs.shape[1] # Récupère les 768 dimensions
    # On utilise IndexFlatL2 (distance Euclidienne)
    index = faiss.IndexFlatL2(dimension)
    index.add(vecteurs)
    return index

def sauvegarder_index(index, chunks_avec_meta: list, chemin: str):
    """Sauvegarde l'index FAISS et les métadonnées sur le disque."""
    # Sauvegarde du fichier binaire .index (les vecteurs)
    faiss.write_index(index, f"{chemin}.index")
    # Sauvegarde du fichier .json (le texte et les infos des films)
    with open(f"{chemin}.json", "w", encoding="utf-8") as f:
        json.dump(chunks_avec_meta, f, ensure_ascii=False, indent=4)

def charger_index(chemin: str):
    """Recharge l'index depuis le disque sans réindexer."""
    index = faiss.read_index(f"{chemin}.index")
    with open(f"{chemin}.json", "r", encoding="utf-8") as f:
        chunks_avec_meta = json.load(f)
    return index, chunks_avec_meta

# --- ACTION : ON LANCE LA SAUVEGARDE MAINTENANT ---

# 1. On crée le moteur de recherche (l'index)
mon_index = creer_index_faiss(matrice_finale)

# 2. On enregistre tout sous le nom "base_films"
sauvegarder_index(mon_index, documents, "base_films")

print(f"💾 Étape 4 réussie ! Fichiers 'base_films.index' et 'base_films.json' créés.")
print(f"✅ Nombre de vecteurs indexés : {mon_index.ntotal}")