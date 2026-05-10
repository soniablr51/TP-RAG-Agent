import os
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Charger la clé depuis le .env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# 1. Test Groq
print("--- Test Groq ---")
client = Groq(api_key=api_key)
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Dis bonjour en une phrase."}]
)
print("✅ Groq OK :", response.choices[0].message.content)

# 2. Test Embeddings
print("\n--- Test Embeddings ---")
# On utilise le modèle de l'image (multilingue)
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
vector = model.encode("Test d'embedding")
print(f"✅ Embedding OK - dimension : {len(vector)}")

# 3. Test FAISS
print("\n--- Test FAISS ---")
# 768 est la taille des vecteurs pour ce modèle précis
index = faiss.IndexFlatL2(768)
index.add(np.array([vector], dtype=np.float32))
print(f"✅ FAISS OK - {index.ntotal} vecteur(s) indexé(s)")