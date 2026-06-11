"""
test_retrieval.py
-----------------
Runs all 5 Evaluation Plan queries against the ChromaDB vector store
and prints retrieved chunks with similarity scores for manual inspection.

Run after vector_store.py has been executed:
    python test_retrieval.py
"""

import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_PATH = "vectors/chroma_db"
COLLECTION  = "skincare_tropical"
MODEL_NAME  = "all-MiniLM-L6-v2"
TOP_K       = 4

# ── Evaluation queries from planning.md ──────────────────────────────────────

EVAL_QUESTIONS = [
    {
        "id": 1,
        "question": (
            "I live in a very humid climate and my face is a grease slick by noon. "
            "What budget morning routine will keep me matte without drying out?"
        ),
        "expected_answer": (
            "Use a gentle water-based/gel cleanser, a lightweight sebum-regulating serum "
            "(like Niacinamide), and skip heavy creams entirely. Finish with a sebum-controlling "
            "chemical or milk-type fluid sunscreen that dries matte."
        ),
        "expected_keywords": ["niacinamide", "gel cleanser", "sunscreen", "matte", "sebum"],
    },
    {
        "id": 2,
        "question": (
            "Are heavy cream moisturizers bad for tropical weather? "
            "What affordable alternatives should a beginner look for?"
        ),
        "expected_answer": (
            "Yes, heavy occlusive creams can trap sweat and sebum in high humidity, causing "
            "breakouts and a greasy texture. Beginners should look for affordable oil-free "
            "water-gels, gel-creams, or aloe/soothing gels (e.g., Holika Holika, Illiyoon, or Isntree)."
        ),
        "expected_keywords": ["gel", "water", "occlusive", "humid", "lightweight"],
    },
    {
        "id": 3,
        "question": (
            "I sweat a lot walking around. "
            "What is a highly-rated, cheap sunscreen that won't melt off or leave a white cast?"
        ),
        "expected_answer": (
            "Look for lightweight, fluid/milk-type sunscreens with matte or sebum-control finishes "
            "(like Biore UV Aqua Rich Watery Essence or Skin Aqua UV Super Moisture Gel). "
            "They are cheap, absorb cleanly, and do not leave heavy white masks."
        ),
        "expected_keywords": ["biore", "sunscreen", "matte", "fluid", "milk", "spf"],
    },
]

# testing Q1, Q2, Q3 only per milestone requirement

# ── Load store ────────────────────────────────────────────────────────────────

def load_store():
    print(f"Loading ChromaDB from {CHROMA_PATH}...")
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION)
    model      = SentenceTransformer(MODEL_NAME)
    print(f"  Collection size: {collection.count()} chunks")
    return collection, model

# ── Retrieve ──────────────────────────────────────────────────────────────────

def retrieve(query, collection, model, top_k=TOP_K):
    embedding = model.encode(query).tolist()
    results   = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "sim":      round(1 - results["distances"][0][i], 4),
        })
    return hits

# ── Relevance check ───────────────────────────────────────────────────────────

def relevance_flag(text, keywords):
    """Check how many expected keywords appear in the retrieved text."""
    text_lower = text.lower()
    matched    = [kw for kw in keywords if kw.lower() in text_lower]
    return matched

# ── Print results ─────────────────────────────────────────────────────────────

def print_query_results(eq, hits):
    print("\n" + "=" * 70)
    print(f"EVAL Q{eq['id']}: {eq['question']}")
    print(f"EXPECTED : {eq.get('expected_answer', '')[:120]}...")
    print("=" * 70)

    for i, hit in enumerate(hits, 1):
        meta    = hit["metadata"]
        matched = relevance_flag(hit["text"], eq["expected_keywords"])
        flag    = "RELEVANT" if matched else "CHECK"

        print(f"\n  [{i}] sim={hit['sim']}  |  {flag}  |  keywords matched: {matched or 'none'}")
        print(f"       source : {meta.get('source')}  ({meta.get('source_type')})")
        print(f"       type   : {meta.get('type')}", end="")
        if meta.get("score") is not None:
            print(f"  |  reddit score: {meta.get('score')}  depth: {meta.get('depth')}", end="")
        print()
        print(f"       url    : {meta.get('url')}")
        print(f"       text   :")
        # wrap text at 65 chars for readability
        words, line = hit["text"].split(), ""
        for word in words:
            if len(line) + len(word) + 1 > 65:
                print(f"         {line}")
                line = word
            else:
                line = (line + " " + word).strip()
        if line:
            print(f"         {line}")

    # summary for this query
    relevant_count = sum(1 for h in hits if relevance_flag(h["text"], eq["expected_keywords"]))
    print(f"\n  SUMMARY: {relevant_count}/{TOP_K} chunks matched expected keywords")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    collection, model = load_store()

    all_results = {}
    for eq in EVAL_QUESTIONS:
        hits = retrieve(eq["question"], collection, model)
        print_query_results(eq, hits)
        all_results[eq["id"]] = hits

    # overall summary table
    print("\n\n" + "=" * 70)
    print("OVERALL RETRIEVAL SUMMARY")
    print("=" * 70)
    print(f"  {'Q':<4} {'Relevant/Total':<18} {'Avg Similarity':<18} {'Top Source'}")
    print(f"  {'-'*4} {'-'*18} {'-'*18} {'-'*20}")
    for eq in EVAL_QUESTIONS:
        hits    = all_results[eq["id"]]
        rel     = sum(1 for h in hits if relevance_flag(h["text"], eq["expected_keywords"]))
        avg_sim = round(sum(h["sim"] for h in hits) / len(hits), 4)
        top_src = hits[0]["metadata"].get("source", "?") if hits else "?"
        print(f"  Q{eq['id']:<3} {str(rel) + '/' + str(TOP_K):<18} {avg_sim:<18} {top_src}")

    print("\n✅ Retrieval test complete")
    print("   Check RELEVANT/CHECK flags above to assess chunk quality.")
    print("   Low sim scores (<0.30) or 'none' keyword matches = retrieval gap.")


if __name__ == "__main__":
    main()