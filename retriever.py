from rank_bm25 import BM25Okapi
import numpy as np
from config import TOP_K

def hybrid_search(question, modele, index, documents, k=TOP_K, poids_bm25=0.3, poids_semantique=0.7):
    try:
        # Recherche sémantique
        vecteur_question = modele.encode([question])
        D, I = index.search(vecteur_question, k * 2)

        scores_semantiques = {}
        for i in range(len(I[0])):
            idx = int(I[0][i])
            distance = float(D[0][i])
            score = 1 / (1 + distance)
            scores_semantiques[idx] = score

        # Recherche BM25
        corpus = [doc["contenu"].lower().split() for doc in documents]
        bm25 = BM25Okapi(corpus)
        tokens_question = question.lower().split()
        scores_bm25_bruts = bm25.get_scores(tokens_question)

        max_bm25 = max(scores_bm25_bruts) if max(scores_bm25_bruts) > 0 else 1
        scores_bm25 = {i: scores_bm25_bruts[i] / max_bm25 for i in range(len(documents))}

        # Fusion des scores
        tous_les_idx = set(scores_semantiques.keys()) | set(
            np.argsort(scores_bm25_bruts)[-k * 2:]
        )

        scores_finaux = {}
        for idx in tous_les_idx:
            score_sem = scores_semantiques.get(idx, 0)
            score_bm = scores_bm25.get(idx, 0)
            scores_finaux[idx] = (poids_semantique * score_sem) + (poids_bm25 * score_bm)

        idx_tries = sorted(scores_finaux, key=scores_finaux.get, reverse=True)[:k]

        resultats = []
        for idx in idx_tries:
            chunk = documents[idx]
            resultats.append({
                "contenu": chunk["contenu"],
                "metadata": chunk["metadata"],
                "score": scores_finaux[idx]
            })

        return resultats

    except Exception as e:
        print(f"Erreur lors de la recherche hybride : {e}")
        return []