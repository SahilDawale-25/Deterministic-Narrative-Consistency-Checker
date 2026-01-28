# src/main.py

import os
import re
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from ingest_books import ingest_and_index_books


# ---------------------------
# SYMBOLIC PATTERNS
# ---------------------------
STATE_PATTERNS = {
    "death": [
        r"\bdied\b", r"\bdead\b", r"fell lifeless", r"killed",
        r"perished", r"slain", r"murdered", r"passed away"
    ],
    "imprisoned": [
        r"prison", r"dungeon", r"cell", r"arrested",
        r"captive", r"locked up", r"detained"
    ]
}

ACTION_PATTERNS = [
    r"travel", r"went", r"spoke", r"married", r"climbed",
    r"joined", r"fought", r"escaped", r"attacked", r"helped"
]

PRONOUNS = ["he", "she", "they", "him", "her"]


# ---------------------------
# UTILITY FUNCTIONS
# ---------------------------
def mentions_character(text, char):
    text_l = text.lower()
    char_l = char.lower()
    return char_l in text_l or any(p in text_l for p in PRONOUNS)


def build_symbolic_memory(chunks, char):
    memory = {"death_positions": [], "imprisoned_positions": []}

    for idx, chunk in enumerate(chunks):
        if not mentions_character(chunk["chunk_text"], char):
            continue

        text_l = chunk["chunk_text"].lower()

        if any(re.search(p, text_l) for p in STATE_PATTERNS["death"]):
            memory["death_positions"].append(idx)

        if any(re.search(p, text_l) for p in STATE_PATTERNS["imprisoned"]):
            memory["imprisoned_positions"].append(idx)

    return memory


def check_causal_violations(memory, chunks):
    contradictions = []

    for idx, chunk in enumerate(chunks):
        text_l = chunk["chunk_text"].lower()

        for action in ACTION_PATTERNS:
            if re.search(action, text_l):
                if memory["death_positions"] and idx > min(memory["death_positions"]):
                    contradictions.append("action_after_death")

                if memory["imprisoned_positions"] and idx > min(memory["imprisoned_positions"]):
                    contradictions.append("action_after_imprisonment")

    return contradictions


def retrieve_top_chunks(claim, vector_store, top_k=5):
    chunks = vector_store["chunks"]
    tfidf_matrix = vector_store["tfidf_matrix"]
    vectorizer = vector_store["vectorizer"]

    claim_vec = vectorizer.transform([claim])
    scores = cosine_similarity(claim_vec, tfidf_matrix).flatten()

    top_indices = scores.argsort()[-top_k:][::-1]

    top_chunks = [chunks[i]["chunk_text"] for i in top_indices]
    top_scores = [scores[i] for i in top_indices]

    return top_chunks, top_scores


# ---------------------------
# MAIN
# ---------------------------
def main():
    data_file = os.path.join("data", "test.csv")
    novels_dir = os.path.join("data", "novels")

    print("📚 Ingesting novels...")
    vector_store = ingest_and_index_books(novels_dir)
    print(f"✅ Total chunks indexed: {len(vector_store['chunks'])}")

    df = pd.read_csv(data_file)
    outputs = []

    print(f"🚀 Processing {len(df)} claims...")

    for _, row in df.iterrows():
        claim_id = row["id"]
        claim = row["content"]
        char = str(row["char"])
        books = row["book_name"].split("|")

        relevant_chunks = [
            c for c in vector_store["chunks"]
            if c["book_name"] in books
        ]

        memory = build_symbolic_memory(relevant_chunks, char)
        contradictions = check_causal_violations(memory, relevant_chunks)

        top_chunks, top_scores = retrieve_top_chunks(claim, vector_store, top_k=5)
        semantic_support = max(top_scores) if top_scores else 0

        # FINAL DECISION
        prediction = 1  # consistent
        if contradictions or (memory["death_positions"] and semantic_support < 0.4):
            prediction = 0  # contradict

        outputs.append({
            "Story_id": claim_id,
            "Prediction": prediction,
            "Rationale": " ".join(top_chunks[:2])  # story-based only
        })

        print(f"✅ ID {claim_id} → {prediction}")

    # SAVE OUTPUT
    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(outputs).to_csv("outputs/results.csv", index=False)

    with open("outputs/results.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)

    print("\n🎯 Done! results.csv saved successfully.")


if __name__ == "__main__":
    main()
