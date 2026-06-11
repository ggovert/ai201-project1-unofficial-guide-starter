"""
app.py
Milestone 5 — Generation and Interface
----------------------------------------
Streamlit Q&A app powered by RAG:
  - User types a skincare question
  - Top-k=4 chunks retrieved from ChromaDB (comments + articles only)
  - Chunks fed into Groq llama-3.3-70b-versatile with strict system prompt
  - If no good chunks found, falls back to general LLM answer (clearly labelled)
  - Answer displayed with source table + text snippet below
  - Low similarity chunks filtered out before generation

Install:
    pip install streamlit chromadb sentence-transformers groq
Run:
    streamlit run app.py
"""

import os
import chromadb
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# load .env file so GROQ_API_KEY is available via os.environ
load_dotenv()

# reuse retrieval functions from vector_store.py — no duplication
from vector_store import retrieve, filter_hits, load_chroma, load_embedding_model, build_vector_store

# ── Auto-build vector store if missing ────────────────────────────────────────
# vectors/ is excluded from git — rebuilt from data/skincare_chunks.json on startup

CHROMA_PATH_CHECK = "vectors/chroma_db"
CHUNKS_PATH       = "data/skincare_chunks.json"

if not os.path.exists(CHROMA_PATH_CHECK):
    import json
    st.toast("Building vector store for the first time — this takes ~1 min...", icon="⚙️")
    with st.spinner("Setting up the database from skincare_chunks.json — please wait..."):
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        build_vector_store(chunks)
    st.success("Database ready!")
    st.rerun()

# ── Config ─────────────────────────────────────────────────────────────────────

CHROMA_PATH    = "vectors/chroma_db"
COLLECTION     = "skincare_tropical"
MODEL_NAME     = "all-MiniLM-L6-v2"
TOP_K          = 4
MIN_SIMILARITY = 0.40
GROQ_MODEL     = "llama-3.3-70b-versatile"

# ── System prompts ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_RAG = """You are a friendly, knowledgeable peer helping skincare beginners
who live in hot, humid tropical climates. You speak like a helpful friend — warm,
clear, and practical — not like a clinical expert.

STRICT RULES you must follow:
1. You can ONLY use information from the SOURCE CHUNKS provided below the question.
2. You must explicitly reference the source (e.g. "According to r/AsianBeauty..."
   or "A Dewsia article explains...") at least once in your answer.
3. If the chunks do not contain enough information to answer confidently, respond
   with exactly: "I don't have enough information from my sources to answer that confidently."
4. Never recommend products, ingredients, or routines that are not mentioned in the chunks.
5. Never make up facts. No hallucination.
6. Keep the answer concise — 3 to 5 sentences max unless a routine list is needed.
7. Write in plain English. Avoid jargon unless you explain it immediately after."""

SYSTEM_PROMPT_FALLBACK = """You are a friendly, knowledgeable peer helping skincare beginners
who live in hot, humid tropical climates. You speak like a helpful friend — warm,
clear, and practical — not like a clinical expert.

The user asked a skincare question that could not be answered from a local database,
so you are answering from your general knowledge. Be helpful and accurate.
Keep the answer concise — 3 to 5 sentences max unless a routine list is needed.
Write in plain English."""

SYSTEM_PROMPT_GUARDRAIL = """You are a topic classifier. 
A user has submitted a question to a skincare and makeup assistant.
Your job is to decide if the question is related to skincare, makeup, 
beauty products, skin health, or cosmetic ingredients.

Reply with ONLY one word:
- ALLOWED  — if the question is about skincare, makeup, beauty, or cosmetic topics
- BLOCKED  — if the question is about anything else (food, politics, coding, sports, etc.)

No explanation. No punctuation. Just one word."""

# load_embedding_model, load_chroma, retrieve, filter_hits
# all imported from vector_store.py above — no duplication

# ── API key helper ─────────────────────────────────────────────────────────────

def get_groq_key() -> str:
    """Read GROQ_API_KEY from st.secrets (Streamlit Cloud) or .env (local)."""
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")

# ── Guardrail ───────────────────────────────────────────────────────────────────

def is_skincare_question(question: str) -> bool:
    """Use Groq to classify if the question is skincare/makeup related."""
    client = Groq(api_key=get_groq_key())
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_GUARDRAIL},
            {"role": "user",   "content": question},
        ],
        temperature=0.0,
        max_tokens=5,
    )
    verdict = response.choices[0].message.content.strip().upper()
    return verdict == "ALLOWED"

# ── Generation ──────────────────────────────────────────────────────────────────

def build_context(hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, 1):
        source = hit["metadata"].get("source", "unknown")
        parts.append(f"[Source {i} — {source}]\n{hit['text']}")
    return "\n\n".join(parts)


def generate_rag_answer(question: str, hits: list[dict]) -> str:
    """Generate answer grounded in retrieved chunks."""
    client  = Groq(api_key=get_groq_key())
    context = build_context(hits)

    user_message = f"""SOURCE CHUNKS:
{context}

QUESTION:
{question}

Answer using only the source chunks above. Reference the sources explicitly."""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.3,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


def generate_fallback_answer(question: str) -> str:
    """Generate a general LLM answer when no good chunks are found."""
    client = Groq(api_key=get_groq_key())

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_FALLBACK},
            {"role": "user",   "content": question},
        ],
        temperature=0.5,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()

# ── Streamlit UI ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tropical Skincare Guide",
    page_icon="🌿",
    layout="centered",
)

st.markdown("## 🌿 Tropical Skincare Guide")
st.markdown(
    "Ask anything about skincare for hot, humid climates. "
    "Answers are grounded in community reviews and expert ingredient guides — no guesswork."
)
st.divider()

# load resources
try:
    embed_model = load_embedding_model()
    collection  = load_chroma()
except Exception as e:
    st.error(f"Could not load vector store. Have you run `vector_store.py` first?\n\n{e}")
    st.stop()

# check groq key early
groq_key = get_groq_key()
if not groq_key:
    st.error("GROQ_API_KEY not set. Add it to your environment variables and restart.")
    st.stop()

# question input
question = st.text_area(
    "Your question",
    placeholder="e.g. What lightweight sunscreen won't melt off in humid weather?",
    height=100,
)

ask_btn = st.button("Ask", type="primary", use_container_width=True)

if ask_btn and question.strip():

    # ── guardrail — block off-topic questions ──────────────────────────────────
    with st.spinner("Checking your question..."):
        allowed = is_skincare_question(question)

    if not allowed:
        st.error(
            "🚫 This assistant only answers questions about skincare and makeup. "
            "Please ask something related to skincare routines, beauty products, "
            "ingredients, or makeup for tropical climates."
        )
        st.stop()

    with st.spinner("Searching our database..."):
        hits          = retrieve(question, collection, embed_model)
        filtered_hits = filter_hits(hits)

    # ── CASE 1: good chunks found — RAG answer ─────────────────────────────────
    if filtered_hits:
        with st.spinner("Generating answer from sources..."):
            answer = generate_rag_answer(question, filtered_hits)

        st.markdown("### Answer")
        st.markdown(answer)
        st.divider()

        # source table
        st.markdown("### Sources used to answer your question")
        st.caption(
            "Match strength shows how closely each source relates to your question "
            "(higher = more relevant)."
        )

        for i, hit in enumerate(filtered_hits, 1):
            meta       = hit["metadata"]
            sim_pct    = round(hit["sim"] * 100, 1)
            source     = meta.get("source", "unknown")
            url        = meta.get("url", "")
            chunk_type = meta.get("type", "")
            snippet    = hit["text"]

            if hit["sim"] >= 0.70:
                strength = "🟢 Strong"
            elif hit["sim"] >= 0.55:
                strength = "🟡 Good"
            else:
                strength = "🟠 Moderate"

            with st.expander(f"Source {i} — {source}  |  {strength}  ({sim_pct}% match)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Type:** {chunk_type}")
                with col2:
                    if meta.get("score") is not None:
                        st.markdown(f"**Community votes:** {meta.get('score')}")
                if url:
                    st.markdown(f"**Link:** [{url}]({url})")
                st.markdown("**Text snippet:**")
                st.info(snippet[:500] + ("..." if len(snippet) > 500 else ""))

    # ── CASE 2: no good chunks — fallback to general LLM ──────────────────────
    else:
        st.info(
            "⚠️ This question wasn't found in our tropical skincare database. "
            "The answer below comes from the AI's general knowledge, not our curated sources."
        )

        with st.spinner("Generating general answer..."):
            fallback_answer = generate_fallback_answer(question)

        st.markdown("### Answer")
        st.markdown(fallback_answer)

        st.caption(
            "💡 This answer is **not** from our database. "
            "For database-backed answers, try asking specifically about "
            "tropical climate skincare, humid weather routines, or budget products for SEA."
        )

elif ask_btn and not question.strip():
    st.warning("Please enter a question first.")