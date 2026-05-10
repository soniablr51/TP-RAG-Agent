# Compte-rendu - TP RAG Films

## Décisions de conception

**Conversion CSV → texte (Q1)** : Chaque ligne du CSV est transformée en texte narratif combinant titre, année, genres, note et synopsis. Ce format permet à l'embedding de capturer le sens global du film plutôt que des champs isolés.

**Extraction des genres (Q2)** : La colonne `genres` est au format JSON imbriqué. J'utilise `json.loads()` pour la parser et extraire uniquement les noms avec une compréhension de liste.

**Persistance de l'index (Q3)** : L'index FAISS et les métadonnées sont sauvegardés sur disque (`base_films.index` + `base_films.json`). `rag.py` les charge au démarrage sans relancer l'indexation.

**Prompt système (Q4)** : Le LLM est contraint de citer titre, note et année pour chaque recommandation, et d'indiquer explicitement si l'information n'est pas dans le contexte.

**Films récents absents (Q5)** : Le dataset date de 2017. Si un film récent est demandé, le LLM signale poliment qu'il ne trouve pas l'information dans sa base plutôt que d'inventer.

**Modèle multilingue** : Choix de `paraphrase-multilingual-mpnet-base-v2` car les questions sont en français mais les synopsis en anglais. Ce modèle gère les deux langues dans le même espace vectoriel.

**Recherche hybride BM25 + sémantique** : Combinaison de BM25 (poids 0.3) et de la recherche vectorielle FAISS (poids 0.7) pour améliorer la pertinence des résultats.

## Difficultés rencontrées

- **Fichiers binaires dans Git** : L'index FAISS (17 MB) et le JSON (3.5 MB) ont été committés par erreur. Résolu avec `git rm --cached` et mise à jour du `.gitignore`.
- **Modèle non multilingue** : Le modèle initial `all-mpnet-base-v2` ne comprenait pas les requêtes en français. Remplacé par le modèle multilingue après avoir constaté des résultats incohérents.
- **Import circulaire** : `orchestrator.py` importait `rag.py` qui importait `orchestrator.py`. Résolu en séparant les responsabilités : l'orchestrateur ne fait que classifier et retourner des réponses fixes.