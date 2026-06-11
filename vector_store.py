"""
vector_store.py
Milestone 4 — Embedding and Retrieval
---------------------------------------
- Loads chunks from data/skincare_chunks.json
- Embeds with all-MiniLM-L6-v2 via sentence-transformers
- Persists ChromaDB collection to vectors/chroma_db/
- Runs a test query returning top-k=4 results with full metadata

Install:
    pip install sentence-transformers chromadb
"""

import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

CHUNKS_PATH    = "data/skincare_chunks.json"
CHROMA_PATH    = "vectors/chroma_db"
COLLECTION     = "skincare_tropical"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K          = 4

# ── Load chunks ───────────────────────────────────────────────────────────────

def load_chunks(path: str) -> list[dict]:
    print(f"Loading chunks from {path}...")
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  Loaded {len(chunks)} chunks")
    reddit  = sum(1 for c in chunks if c["source_type"] == "reddit")
    website = sum(1 for c in chunks if c["source_type"] == "website")
    print(f"  Reddit: {reddit}  |  Website: {website}")
    return chunks

# ── Shared loader functions (importable by app.py) ───────────────────────────

def load_embedding_model():
    """Load and return the sentence-transformers embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL)


def load_chroma():
    """Load and return the existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(COLLECTION)


def filter_hits(hits: list[dict], min_sim: float = 0.40) -> list[dict]:
    """Filter out chunks below the minimum similarity threshold."""
    return [h for h in hits if h["sim"] >= min_sim]


# ── Build ChromaDB collection ─────────────────────────────────────────────────

def build_vector_store(chunks: list[dict]) -> chromadb.Collection:
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Creating ChromaDB at {CHROMA_PATH}...")
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # drop and recreate so re-runs stay clean
    try:
        client.delete_collection(COLLECTION)
        print(f"  Dropped existing '{COLLECTION}' collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    # embed in batches of 64 for memory efficiency
    BATCH = 64
    total = len(chunks)
    print(f"\nEmbedding {total} chunks in batches of {BATCH}...")

    for i in range(0, total, BATCH):
        batch = chunks[i : i + BATCH]

        texts = [c["text"] for c in batch]
        ids   = [f"chunk_{i + j}" for j in range(len(batch))]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # build metadata — only include non-None values ChromaDB can store
        metadatas = []
        for c in batch:
            meta = {
                "source_type": c.get("source_type", ""),
                "source":      c.get("source", ""),
                "url":         c.get("url", ""),
                "type":        c.get("type", ""),
            }
            # reddit-only fields
            if c.get("score") is not None:
                meta["score"] = int(c["score"])
            if c.get("depth") is not None:
                meta["depth"] = int(c["depth"])
            if c.get("date") is not None:
                meta["date"] = str(c["date"])
            # website-only fields
            if c.get("chunk_index") is not None:
                meta["chunk_index"] = int(c["chunk_index"])
            if c.get("chunk_total") is not None:
                meta["chunk_total"] = int(c["chunk_total"])
            metadatas.append(meta)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        done = min(i + BATCH, total)
        print(f"  Embedded {done}/{total} chunks", end="\r")

    print(f"\n  Done. Collection size: {collection.count()} chunks")
    return collection, model

# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(query: str, collection: chromadb.Collection, model: SentenceTransformer, top_k: int = TOP_K) -> list[dict]:
    """Embed query and return top-k most similar chunks with metadata."""
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
        where={"type": {"$in": ["top_level_comment", "reply", "article"]}},
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "sim":      round(1 - results["distances"][0][i], 4),  # cosine similarity
        })
    return hits


def print_results(query: str, hits: list[dict]) -> None:
    print("\n" + "=" * 65)
    print(f"Query: {query}")
    print("=" * 65)
    for i, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        print(f"\n[{i}] similarity : {hit['sim']}")
        print(f"    source     : {meta.get('source')}  ({meta.get('source_type')})")
        print(f"    type       : {meta.get('type')}")
        print(f"    url        : {meta.get('url')}")
        if meta.get("score") is not None:
            print(f"    reddit score : {meta.get('score')}  |  depth: {meta.get('depth')}")
        print(f"    text preview :")
        print(f"    {hit['text'][:300]}...")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. load
    chunks = load_chunks(CHUNKS_PATH)

    # 2. embed + store
    collection, model = build_vector_store(chunks)

    # 3. test queries — Evaluation Questions Q1, Q2, Q3
    queries = [
        (
            "I live in a very humid climate and my face is a grease slick by noon. "
            "What budget morning routine will keep me matte without drying out?"
        ),
        (
            "Are heavy cream moisturizers bad for tropical weather? "
            "What affordable alternatives should a beginner look for?"
        ),
        (
            "I sweat a lot walking around. "
            "What is a highly-rated, cheap sunscreen that won't melt off or leave a white cast?"
        ),
    ]

    for query in queries:
        hits = retrieve(query, collection, model)
        print_results(query, hits)

    print("\n✅ vector_store.py complete — ChromaDB persisted to", CHROMA_PATH)


if __name__ == "__main__":
    main()