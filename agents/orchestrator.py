SALUTATIONS = ["bonjour", "salut", "hello", "bonsoir", "hey", "coucou"]
AIDE = ["aide", "help", "que peux-tu faire", "comment ça marche", "quoi faire"]

def classify_query(question: str) -> str:
    question_lower = question.lower().strip()
    if any(s in question_lower for s in SALUTATIONS):
        return "greeting"
    if any(a in question_lower for a in AIDE):
        return "help"
    return "rag"

def handle_greeting() -> str:
    return "Bonjour ! Je suis votre assistant cinéma. Posez-moi une question sur un film ou demandez-moi une recommandation !"

def handle_help() -> str:
    return """Je peux vous aider à :
- Trouver des films par genre, thème ou ambiance
- Vous donner des informations sur un film précis
- Recommander des films similaires

Exemples de questions :
- "Recommande-moi un film d'action des années 90"
- "Quel est le meilleur film de science-fiction ?"
- Ajoutez ' FR' à la fin pour filtrer les films français
- Ajoutez ' VO' pour les films en anglais"""