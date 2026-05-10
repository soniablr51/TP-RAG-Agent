# RAG Agent - Recommandations de Films

Système de recommandation de films basé sur RAG (Retrieval-Augmented Generation) utilisant le dataset TMDB 5000.

## Prérequis

- Python 3.10+
- Une clé API Groq : https://console.groq.com

## Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-username>/RAG-Agent-TP.git
cd RAG-Agent-TP

# 2. Créer et activer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Configurer la clé API
# Créer un fichier .env à la racine avec :
GROQ_API_KEY=votre_clé_ici
```

## Lancer le projet

```bash
# Étape 1 : Indexer les films (à faire une seule fois)
python indexation.py

# Étape 2 : Lancer l'agent
python rag.py
```

## Exemples de questions

```
Recommande-moi un film de science-fiction
Quel est le meilleur film d'aventure ?
Recommande-moi un film FR       ← films français uniquement
Recommande-moi un film VO       ← films en anglais uniquement
bonjour                         ← message d'accueil
aide                            ← liste des fonctionnalités
```

## Structure du projet

```
RAG-Agent-TP/
├── agents/
│   ├── __init__.py
│   └── orchestrator.py   # Routage des intentions
├── config.py             # Constantes centralisées
├── indexation.py         # Création de la base FAISS
├── retriever.py          # Recherche hybride BM25 + sémantique
├── rag.py                # Pipeline principal
└── requirements.txt
```

## Technologies

- **Embeddings** : `paraphrase-multilingual-mpnet-base-v2` (multilingue)
- **Base vectorielle** : FAISS
- **Recherche hybride** : BM25 + sémantique (30/70)
- **LLM** : Llama 3.3 70B via Groq API
- **Dataset** : TMDB 5000 Movies



## Choix techniques (Questions de réflexion)

**Q1 - Conversion CSV → texte** : Chaque film est converti en texte narratif combinant titre, année, genres, note et synopsis. Ce format permet à l'embedding de capturer le sens global du film.

**Q2 - Extraction des genres** : La colonne `genres` est au format JSON imbriqué. J'utilise `json.loads()` pour parser et extraire uniquement les noms des genres.

**Q3 - Persistance de l'index** : L'index FAISS et les métadonnées sont sauvegardés sur disque. `rag.py` les charge au démarrage sans relancer l'indexation.

**Q4 - Prompt système** : Le LLM est contraint de citer titre, note et année pour chaque recommandation, et de signaler si l'information n'est pas dans le contexte.

**Q5 - Films récents absents** : Le dataset date de 2017. Si un film récent est demandé, le LLM l'indique poliment plutôt que d'inventer.