# The Unofficial Guide — Project 1
---

## Domain
This system serves as an unofficial guide for skincare beginners living in hot, humid tropical climates who are on a budget. Standard skincare advice often recommends heavy, expensive products that clog pores or melt off in extreme humidity and sweat. By aggregating community-vetted threads and ingredient guides focused on lightweight, affordable formulations (like gels and fluid sunscreens), this tool helps users build an effective, sweat-proof routine without overspending.

---

## Document Sources

| # | Source | Type | Description | URL or file path |
|---|--------|------|-------------|------------------|
| 1 | 1883 Magazine | Website | The Beauty Editor's Guide to What to Buy When Travelling to Asia | https://1883magazine.com/the-beauty-editors-guide-to-what-to-buy-when-travelling-to-asia/ |
| 2 | V10 Plus Blog | Website | Skin Care Secrets for Thriving in Southeast Asia's Tropical Climate | https://v10plus.com/blogs/news/skin-care-secrets-for-thriving-in-southeast-asias-tropical-climate |
| 3 | Dewsia Article | Website | Ultimate Skincare Routine for Hot and Humid Tropical Weather | https://dewsia.com/skincare-routine-tropical-weather/ |
| 4 | r/AsianBeauty | Reddit | Best & HG Products for Tropical/humid climates | https://www.reddit.com/r/AsianBeauty/comments/1cffssa/best_hg_products_for_tropicalhumid_climates/ |
| 5 | r/SkincareAddiction | Reddit | Routine Help: How humid weather completely changed my routine | https://www.reddit.com/r/SkincareAddiction/comments/1otyffj/routine_help_how_humid_weather_completely_changed/ |
| 6 | r/AsianBeauty | Reddit | Oily/combo acne prone, sensitive, humid climate routines | https://www.reddit.com/r/AsianBeauty/comments/1ryi2ea/oily_combo_acne_prone_sensitive_humid_climate/ |
| 7 | r/AsianBeauty | Reddit | Day and night moisturiser choices for tropical climates | https://www.reddit.com/r/AsianBeauty/comments/xuezv1/for_those_living_in_tropical_climates_what_day/ |
| 8 | r/AsianBeauty | Reddit | Massive community list of budget HG products | https://www.reddit.com/r/AsianBeauty/comments/14jrgkc/what_are_your_budget_hg_products/ |
| 9 | IncideCoder Wiki | Website | Niacinamide: Sebum regulation & barrier repair mechanics | https://incidecoder.com/ingredients/niacinamide |
| 10 | IncideCoder Wiki | Website | Salicylic Acid / BHA: Lipophilic pore clearing properties | https://incidecoder.com/ingredients/salicylic-acid |
| 11 | IncideCoder Wiki | Website | Dimethicone: Lightweight silicone barriers vs heavy occlusives | https://incidecoder.com/ingredients/dimethicone |

---

## Chunking Strategy

**Chunk size:**
- Reddit (via `.json` scraping): 1 comment = 1 chunk (no fixed token split)
- Website (scraped articles/pages): 300–500 tokens per chunk

**Overlap:**
- Reddit: Parent comment first 150 characters prepended to reply chunks
- Website: 75 token overlap between chunks

**Reddit (direct `.json` + recursive traversal):**
Reddit's self-service API (PRAW) was closed to new developers in 2025.
Instead, Reddit data is fetched by appending `.json` to any post URL
and calling it via `requests` with a descriptive User-Agent header.
This returns the same structured comment tree that PRAW used internally —
each comment arrives as a clean JSON object with score, depth, and
created_utc already attached. No credentials or API key are required.

Each comment is treated as one atomic chunk. Recursive depth-first
traversal walks the tree up to depth 2. Top-level comments (depth 0)
are chunked standalone. Replies (depth 1) prepend the first 150
characters of their parent comment so the chunk remains meaningful
in isolation. Traversal stops at depth 2 to avoid off-topic banter.
No token splitting is needed since Reddit comments naturally average
80–200 tokens.

**Website (requests + BeautifulSoup):**
Website content is long-form HTML — articles, ingredient guides,
product descriptions — with no natural atomic boundary. `requests`
fetches the raw HTML and BeautifulSoup strips noise tags (nav, footer,
script, style) before extracting clean body text. That text is then
split using RecursiveCharacterTextSplitter at 300–500 tokens with
75 token overlap. Chunking is token-based (tiktoken cl100k_base) to
stay predictable for the embedding model.

**Why `.json` for Reddit instead of BeautifulSoup:**
BeautifulSoup parses raw HTML — it would require fragile CSS selectors
to extract comment text, scores, and thread depth from Reddit's page
structure, which changes frequently. Reddit's `.json` endpoint returns
pre-structured data with all metadata already attached, making it
faster, more reliable, and simpler to maintain. BeautifulSoup is
reserved for external websites where no structured API exists.


**Why these choices fit your documents:**
Two sources are used — Reddit and skincare websites — each with
different document structures, so chunking is handled per source.

**Final chunk count:**
- 182 chunks
- Reddit: 141  |  Website: 41
---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via sentence-transformers (local, no API key needed)

This model was chosen for its balance of speed, size, and semantic
quality for a student project. It runs entirely locally, requires no
API calls, and produces 384-dimensional embeddings fast enough to
embed hundreds of Reddit comments in under a minute on CPU. Its
lightweight size (80MB) also makes it practical for Streamlit Cloud's
free tier where memory is limited.


**Production tradeoff reflection:**

If deploying for real users without cost constraints, the main
tradeoffs to weigh are:

- **Context length:** `all-MiniLM-L6-v2` has a hard 256-token input
  limit, which forces aggressive chunking. A production model like
  OpenAI's `text-embedding-3-large` (8191 tokens) or Cohere's
  `embed-english-v3.0` would handle longer Reddit threads and full
  ingredient guides as single chunks, preserving more context per
  retrieval hit.

- **Multilingual support:** Regional beauty subreddits (r/AsianBeauty,
  r/IndianSkincareAddicts) contain Malay, Tagalog, Hindi, and mixed-
  language posts. `all-MiniLM-L6-v2` is English-only and silently
  produces weak embeddings for these. A multilingual model like
  `paraphrase-multilingual-MiniLM-L12-v2` or Cohere's multilingual
  embed would handle this correctly.

- **Domain specificity:** Skincare text contains chemical ingredient
  names (ethylhexyl methoxycinnamate, niacinamide, dimethicone) that
  general-purpose models may not distinguish well semantically. A
  domain-fine-tuned model or one with a larger vocabulary like
  `text-embedding-3-large` would produce more accurate similarity
  scores between ingredient-heavy queries and chunks.

- **Latency vs. accuracy:** Local models like `all-MiniLM-L6-v2` have
  near-zero network latency but lower accuracy ceilings. API-hosted
  models like OpenAI or Cohere are more accurate but add 100–300ms
  per query and introduce an external dependency. For a real-time
  skincare assistant, a hybrid approach — local model for fast
  filtering, API model for final re-ranking — would be the
  production-grade choice.

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**

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

**How source attribution is surfaced in the response:**
Grounding is enforced at three levels:

First, structurally — each chunk is labelled before injection:
[Source 1 — r/AsianBeauty]
<chunk text>
[Source 2 — Dewsia Article]
<chunk text>
This gives the model explicit source names to reference in its
answer rather than relying on it to invent attribution.

Second, through similarity filtering — chunks below a 0.40
cosine similarity threshold are discarded before generation.
This means the model never sees weakly-matched chunks that could
cause it to drift off-topic or hallucinate connections.

Third, through the fallback path — if no chunks pass the 0.40
threshold, the app bypasses the RAG system prompt entirely and
switches to a separate fallback prompt that does not claim source
grounding, displaying a clear UI warning:
*"This answer comes from the AI's general knowledge, not our
curated sources."* This prevents the model from being asked to
cite sources it was never given.

---
## Evaluation Report

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | I live in a very humid climate and my face is a grease slick by noon. What budget morning routine will keep me matte without drying out? | Gentle gel cleanser, lightweight niacinamide serum, skip heavy creams, finish with matte fluid sunscreen. | Recommended mattifying products with niacinamide and salicylic acid, cited V10 Plus Blog and Dewsia. Suggested skipping moisturizer entirely — partially misaligned with expected answer which says skip heavy creams, not all moisturizer. Sunscreen step was included correctly. | Relevant | Partially accurate |
| 2 | Are heavy cream moisturizers bad for tropical weather? What affordable alternatives should a beginner look for? | Yes, heavy occlusives trap sweat and sebum. Look for oil-free water-gels, gel-creams, or aloe gels (Holika Holika, Illiyoon, Isntree). | Correctly identified heavy creams as problematic and explained the humidity/occlusion logic. Recommended COSRX Oil-Free lotion and Heimish Matcha Gel — both valid gel-cream alternatives. Did not mention Holika Holika or Illiyoon from expected answer but the category of alternatives was correct. | Relevant | Accurate |
| 3 | I sweat a lot walking around. What is a highly-rated, cheap sunscreen that won't melt off or leave a white cast? | Biore UV Aqua Rich or Skin Aqua UV Super Moisture Gel — lightweight, matte, cheap, no white cast. | Mentioned Skin Aqua correctly. Also recommended Cocoon Winter Melon Sun Fluid and Beauty of Joseon Relief Sun — both valid community-recommended options. Hedged with "may not leave a white cast" instead of being definitive, which slightly weakens the answer. | Relevant | Accurate |
| 4 | My skin feels tight but looks oily (dehydrated oily). How do I fix this using cheap products available in tropical regions? | Use lightweight humectants like Hyaluronic Acid or watery toners (Hada Labo Lotion) to hydrate without adding oil. | Correctly identified the dehydrated-oily pattern and recommended HA toners (Isntree, Torriden) which match the expected approach. Also mentioned Cosrx snail cream as an option — not in expected answer but not incorrect. Did not mention Hada Labo specifically but the mechanism (lightweight humectants) was right. | Relevant | Accurate |
| 5 | How often should a beginner use a BHA exfoliant to clear out sweat-clogged pores without damaging their skin barrier in hot weather? | Start with Salicylic Acid / BHA 2–3 times per week at night only. Over-exfoliating causes oily rebound. | Retrieved IncideCoder salicylic acid info and a Dewsia article on physical exfoliation, but neither chunk contained specific BHA frequency guidance. System correctly triggered the "I don't have enough information" fallback rather than hallucinating a frequency. | Partially relevant | Partially accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

**Summary observations:**

Q1–Q4 retrieval was consistently relevant, pulling from the correct
Reddit and website sources. Q5 exposed a genuine gap in the corpus —
the ingested sources cover what BHA does but not how often to use it
in tropical conditions specifically, which caused a correct but
incomplete fallback response.

The most notable inaccuracy was in Q1 where the system recommended
skipping moisturizer entirely rather than just skipping heavy creams.
This is a meaningful distinction for beginners — dehydrated oily skin
still needs lightweight hydration. This suggests the system prompt
grounding rules could be tightened to prevent the model from
over-generalising from a single chunk.

Q5's partial retrieval is the strongest signal for corpus improvement
— adding sources specifically covering BHA/AHA usage frequency in
tropical or hot climates would directly close this gap.

---
## Failure Case Analysis

**Question that failed:**
I live in a very humid climate and my face is a grease slick by noon.
What budget morning routine will keep me matte without drying out?

**What the system returned:**
The answer was mostly correct, but one of the retrieved chunks used
as a source was a Reddit post body — specifically the original
question post from r/AsianBeauty asking the community for tropical
skincare recommendations. This chunk contained keywords like
"humid", "tropical", "skincare", and "routine" which produced a
high cosine similarity score against the query, even though the
chunk contained no actual advice — it was itself a question, not
an answer.

**Root cause (tied to a specific pipeline stage):**
The failure originates at two pipeline stages. First, at ingestion
— Reddit post bodies were chunked and stored alongside comment
chunks without distinguishing between question-asking posts and
answer-giving comments. A post that says "Would love to hear what
products work for humid climates" and a comment that says "Use
niacinamide in humid climates" will produce nearly identical
embeddings because they share the same domain vocabulary, even
though one has zero informational value for generation.

Second, at retrieval — the `where` filter in ChromaDB was set to
exclude `type: post` chunks, but this filter was only applied
after the vector store was already built with post chunks indexed.
The depth `-1` metadata tag correctly identifies posts, but the
semantic overlap between question posts and skincare queries means
the embedding model (`all-MiniLM-L6-v2`) cannot distinguish
intent — it only measures topical similarity, not whether a chunk
is a question or an answer. A 256-token context window also means
the model cannot see enough surrounding context to infer that a
chunk is a question rather than a recommendation.

**What you would change to fix it:**
I would develop a better chunking method specifically tailored for Reddit data. 
Not every comment contains the right answer; some are off-topic or add noise. Refining the scraping strategy and text-chunking logic would ensure we extract and process only the most relevant context.

---

## Spec Reflection

**One way the spec helped you during implementation:**
The planning document forced early decisions about chunking
strategy before writing a single line of code. When the PRAW API
became unavailable mid-project, having the chunking logic already
documented in `planning.md` meant the switch to the `.json`
endpoint only required changing the data fetching layer — the
recursive depth-first traversal, parent context prepending, and
metadata tagging were already fully specified and transferred
unchanged. Without the spec, this pivot would have required
rethinking the entire ingestion design under pressure. The spec
also served as a precise prompt context for AI tools — providing
Claude with the full `Chunking Strategy` and `Documents` sections
produced significantly more accurate code than open-ended
requests, because the AI had the exact data structure and
requirements already written out.

**One way your implementation diverged from the spec, and why:**
The original spec specified PRAW as the Reddit ingestion method,
but Reddit closed self-service API access to new developers in
2025, making PRAW unavailable. The implementation pivoted to
direct `.json` endpoint scraping via `requests` and a persistent
session to bypass Reddit's bot detection. This divergence actually
improved the architecture — removing the PRAW dependency meant
no API credentials are required to run the project locally, and
the `.json` endpoint returns the same structured comment tree that
PRAW used internally, so the chunking logic required no changes.
The spec was updated in `planning.md` to document this change,
including a new Anticipated Challenge entry covering Reddit rate
limiting and the `time.sleep()` mitigation strategy.

---

## AI Usage

**Instance 1**
- *What I gave the AI:* Asked Claude to write a Reddit scraper,
  providing the Documents and Chunking Strategy sections from
  `planning.md` as context, specifying PRAW as the tool.
- *What it produced:* A complete `ingest.py` using PRAW with
  recursive comment traversal and metadata tagging — but PRAW
  required API credentials that Reddit no longer issues to new
  developers, so the script failed at authentication.
- *What I changed or overrode:* Provided Claude with Reddit's
  official policy change announcement and documentation about the
  `.json` endpoint as a credential-free alternative. After
  iterating through rate-limiting issues (403 errors), Claude
  produced a session-based approach using `old.reddit.com` with
  browser-spoofed headers and retry backoff logic that
  successfully bypassed Reddit's bot detection.

**Instance 2**
- *What I gave the AI:* Asked Claude to build a Streamlit UI,
  providing the complete `planning.md` and specifying the exact
  sections required: a question input, a grounded answer output,
  a source table with match strength labels, a fallback path for
  off-topic questions, and a guardrail classifier.
- *What it produced:* A near-complete `app.py` with all requested
  sections implemented — the RAG answer path, fallback LLM path,
  ChromaDB source table with expandable cards, cosine similarity
  displayed as plain-language match strength percentages, and a
  Groq-powered topic guardrail. Minor visual adjustments to
  colour labels and card layout were made to better fit the
  project's aesthetic.
- *What I changed or overrode:* Adjusted the similarity threshold
  from the initial value and refined the guardrail system prompt
  after testing showed it occasionally blocked valid skincare
  questions that used informal language.
