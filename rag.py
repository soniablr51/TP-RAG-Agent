from dotenv import load_dotenv
import os
import faiss
import json
from sentence_transformers import SentenceTransformer
from groq import Groq
from config import MODEL_NAME, TOP_K, INDEX_PATH
from retriever import hybrid_search
from agents.orchestrator import classify_query, handle_greeting, handle_help

# Charger les variables du fichier .env
load_dotenv()

# Récupérer la clé API et la mettre dans une variable
api_key = os.getenv("GROQ_API_KEY")

# Vérification (pour toi)
if api_key:
    print(f"Succès ! Ta clé commence par : {api_key[:7]}...")
else:
    print("Erreur : Impossible de lire la clé. Vérifie le nom du fichier .env")

client = Groq(api_key=api_key)

# On charge le modèle pour transformer la question en chiffres
modele = SentenceTransformer(MODEL_NAME)

# On charge l'index FAISS (les vecteurs des 5653 films)
index = faiss.read_index(f"{INDEX_PATH}.index")

# On charge les textes et métadonnées correspondantes
with open(f"{INDEX_PATH}.json", "r", encoding="utf-8") as f:
    documents = json.load(f)



# --- ÉTAPE 6 : GÉNÉRATION AVEC GROQ ---

def construire_prompt_systeme() -> str:
    """Retourne le prompt système pour définir le comportement de l'IA."""
    return """Tu es un assistant expert en cinéma.
Ton rôle est d'aider l'utilisateur en te basant EXCLUSIVEMENT sur le contexte fourni.

RÈGLES :
 1. Si la réponse n'est pas dans le contexte, dis-le poliment. N'invente rien.
 2. Cite toujours le titre du film, sa note /10 et son année de sortie.
 3. Sois concis et amical.
 4. Réponds toujours en français."""

def generer_reponse(question: str, chunks_pertinents: list[dict]) -> str:
    """Assemble le contexte et appelle l'API Groq."""
    
    # On prépare le contexte en listant les films trouvés
    contexte = ""
    for i, res in enumerate(chunks_pertinents, 1):
        titre = res['metadata'].get('titre') or res['metadata'].get('title') or "Inconnu"
        contexte += f"\nSource {i} (Film: {titre}) : {res['contenu']}\n"

    # Appel au modèle Llama 3
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Le modèle performant
            messages=[
                {"role": "system", "content": construire_prompt_systeme()},
                {"role": "user", "content": f"CONTEXTE :\n{contexte}\n\nQUESTION : {question}"}
            ],
            temperature=0.2 # On reste factuel
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Erreur lors de la génération : {e}"

def main():
    print("\n" + "="*50)
    print("🎬 BIENVENUE SUR VOTRE AGENT CINÉMA RAG")
    print("Tapez votre question ou 'q' pour quitter.")
    print("="*50)

    while True:
        # On récupère la question de l'utilisateur
        question = input("\n👤 Votre question : ").strip()

        # Conditions de sortie
        if question.lower() in ["q", "quit", "exit"]:
            print("👋 Au revoir !")
            break

        if not question:
            continue

        intent = classify_query(question)
        if intent == "greeting":
            print(f"\n{'✨' + '-'*48}")
            print(handle_greeting())
            continue
        if intent == "help":
            print(f"\n{'✨' + '-'*48}")
            print(handle_help())
            continue


        # Filtre de langue
        filtre_langue = None
        if question.upper().endswith(" FR"):
            filtre_langue = "fr"
            question = question[:-3].strip()
        elif question.upper().endswith(" VO"):
            filtre_langue = "en"
            question = question[:-3].strip()

        print("🔍 Recherche en cours...")
        
        # 1. On cherche les films dans la base FAISS
        resultats_recherche = hybrid_search(question, modele, index, documents)

        # Appliquer le filtre de langue si demandé
        if filtre_langue:
            resultats_recherche = [
                r for r in resultats_recherche
                if r['metadata'].get('langue') == filtre_langue
            ]
            if not resultats_recherche:
                print(f"Aucun film trouvé en langue '{filtre_langue}'.")
                continue

        print("L'IA prépare votre recommandation...")
        
        # 2. On génère la réponse rédigée
        reponse = generer_reponse(question, resultats_recherche)

        # 3. Affichage du résultat final
        print("\n" + "✨" + "-"*48)
        print(f"CONSEIL DE L'IA :\n\n{reponse}")
        
        print("\n📚 FILMS UTILISÉS POUR CETTE RÉPONSE :")
        for i, res in enumerate(resultats_recherche, 1):
            titre = res['metadata'].get('titre') or res['metadata'].get('title') or "Inconnu"
            print(f"   {i}. {titre} (Score : {res['score']:.4f})")
        print("-"*50)

if __name__ == "__main__":
    main()