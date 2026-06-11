"""
ingest.py
Milestone 3 - Ingestion and Chunking
-------------------------------------
Sources:
  - Reddit  : direct .json endpoint (no API key needed)
               requests + recursive depth-first comment traversal
  - Websites: requests + BeautifulSoup (HTML scraping)

Chunking strategy:
  - Reddit  : 1 comment = 1 chunk, recursive depth-first (max depth 2)
               reply chunks prepend first 150 chars of parent comment
  - Website : RecursiveCharacterTextSplitter, 300-500 tokens, 75 token overlap
               token length via tiktoken cl100k_base

Output: data/skincare_chunks.json
"""

import html
import json
import os
import random
import re
import time

import requests
import tiktoken
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Constants ──────────────────────────────────────────────────────────────────

REDDIT_POSTS = [
    {
        "url": "https://old.reddit.com/r/AsianBeauty/comments/1cffssa/best_hg_products_for_tropicalhumid_climates/",
        "subreddit": "r/AsianBeauty",
    },
    {
        "url": "https://old.reddit.com/r/SkincareAddiction/comments/1otyffj/routine_help_how_humid_weather_completely_changed/",
        "subreddit": "r/SkincareAddiction",
    },
    {
        "url": "https://old.reddit.com/r/AsianBeauty/comments/1ryi2ea/oily_combo_acne_prone_sensitive_humid_climate/",
        "subreddit": "r/AsianBeauty",
    },
    {
        "url": "https://old.reddit.com/r/AsianBeauty/comments/xuezv1/for_those_living_in_tropical_climates_what_day/",
        "subreddit": "r/AsianBeauty",
    },
    {
        "url": "https://old.reddit.com/r/AsianBeauty/comments/14jrgkc/what_are_your_budget_hg_products/",
        "subreddit": "r/AsianBeauty",
    },
]

WEBSITE_SOURCES = [
    {
        "url": "https://1883magazine.com/the-beauty-editors-guide-to-what-to-buy-when-travelling-to-asia/",
        "name": "1883 Magazine",
    },
    {
        "url": "https://v10plus.com/blogs/news/skin-care-secrets-for-thriving-in-southeast-asias-tropical-climate",
        "name": "V10 Plus Blog",
    },
    {
        "url": "https://dewsia.com/skincare-routine-tropical-weather/",
        "name": "Dewsia Article",
    },
    {
        "url": "https://incidecoder.com/ingredients/niacinamide",
        "name": "IncideCoder - Niacinamide",
    },
    {
        "url": "https://incidecoder.com/ingredients/salicylic-acid",
        "name": "IncideCoder - Salicylic Acid",
    },
    {
        "url": "https://incidecoder.com/ingredients/dimethicone",
        "name": "IncideCoder - Dimethicone",
    },
]

MIN_SCORE     = 1    # only drops actively downvoted comments
MAX_DEPTH     = 2
CHUNK_SIZE    = 300  # tokens
CHUNK_OVERLAP = 75   # tokens

REDDIT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

WEBSITE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# ── Tokenizer ──────────────────────────────────────────────────────────────────

TOKENIZER = tiktoken.get_encoding("cl100k_base")

def token_len(text):
    return len(TOKENIZER.encode(text))

# ── Text cleaners ──────────────────────────────────────────────────────────────

def clean_reddit_text(text):
    """Decode HTML entities, strip markdown noise, convert newlines to spaces."""
    text = html.unescape(text)                                  # &amp; -> &
    text = re.sub(r'\*{1,2}(.*?)\*{1,2}', r'\1', text)        # **bold** -> bold
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)      # [text](url) -> text
    text = re.sub(r'http\S+', '', text)                        # remove bare URLs
    text = re.sub(r'\n', ' ', text)                            # newlines -> spaces
    text = re.sub(r' {2,}', ' ', text)                         # multi spaces -> 1
    return text.strip()
 
 
def clean_website_text(text):
    """Collapse newlines into spaces so chunks read as clean prose."""
    text = re.sub(r'\n{3,}', '\n\n', text)                     # 3+ newlines -> 2
    text = re.sub(r'\n', ' ', text)                            # newlines -> space
    text = re.sub(r' {2,}', ' ', text)                         # multi spaces -> 1
    return text.strip()

# ── Reddit session ─────────────────────────────────────────────────────────────

REDDIT_SESSION = requests.Session()
REDDIT_SESSION.headers.update(REDDIT_HEADERS)

def init_reddit_session():
    """
    Warm up the session by visiting Reddit pages before any .json fetches.
    Multiple visits simulate real browser behaviour and populate cookies.
    """
    print("  Initialising Reddit session (warming up cookies)...")
    try:
        REDDIT_SESSION.get("https://old.reddit.com", timeout=15)
        time.sleep(3)
        REDDIT_SESSION.get("https://old.reddit.com/r/AsianBeauty", timeout=15)
        time.sleep(3)
        print("  Session ready.")
    except Exception as e:
        print(f"  Warning: could not init session: {e}")


def fetch_reddit_json(url, retries=3):
    """
    Append .json to a Reddit post URL and fetch structured data.
    Retries with backoff on 403/429 — Reddit bot detection is inconsistent.
    """
    json_url = url.rstrip("/") + ".json"
    for attempt in range(retries):
        try:
            resp = REDDIT_SESSION.get(json_url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  Retry {attempt + 1}/{retries} in {wait}s -- {e}")
                time.sleep(wait)
            else:
                raise

# ── Reddit comment parsing ─────────────────────────────────────────────────────

def parse_comment(comment_data):
    """Extract and validate a comment from raw Reddit JSON. Returns None if skippable."""
    data = comment_data.get("data", {})
    kind = comment_data.get("kind", "")

    if kind == "more":
        return None

    body  = data.get("body", "")
    score = data.get("score", 0)

    if not body or body in ("[deleted]", "[removed]"):
        return None
    if score < MIN_SCORE:
        return None
    if len(body.split()) < 10:
        return None

    return {
        "body":  body.strip(),
        "score": score,
        "date":  str(int(data.get("created_utc", 0))),
        "replies": (
            data.get("replies", {})
                .get("data", {})
                .get("children", [])
            if isinstance(data.get("replies"), dict)
            else []
        ),
    }


def chunk_comment_tree(comment_data, parent_text, chunks, subreddit, post_url, current_depth=0):
    """
    Recursively walk comment tree depth-first.
    depth 0  -> standalone chunk
    depth 1  -> prepend first 150 chars of parent for context
    depth 2+ -> stop
    """
    if current_depth > MAX_DEPTH:
        return

    comment = parse_comment(comment_data)
    if comment is None:
        return

    clean_body   = clean_reddit_text(comment["body"])
    clean_parent = clean_reddit_text(parent_text) if parent_text else None

    if clean_parent:
        chunk_text = (
            f"[Parent context]: {clean_parent[:150].strip()}... "
            f"[Reply]: {clean_body}"
        )
    else:
        chunk_text = clean_body

    chunks.append({
        "text":        chunk_text,
        "source_type": "reddit",
        "source":      subreddit,
        "url":         post_url,
        "score":       comment["score"],
        "depth":       current_depth,
        "date":        comment["date"],
        "type":        "reply" if parent_text else "top_level_comment",
    })

    for reply in comment["replies"]:
        chunk_comment_tree(
            reply,
            parent_text=comment["body"],
            chunks=chunks,
            subreddit=subreddit,
            post_url=post_url,
            current_depth=current_depth + 1,
        )


def scrape_reddit(posts):
    chunks = []
    init_reddit_session()

    for post_meta in posts:
        url      = post_meta["url"]
        sub_name = post_meta["subreddit"]
        post_id  = url.split("comments/")[1].split("/")[0]
        print(f"  Fetching {sub_name} -- {post_id}...")

        try:
            data         = fetch_reddit_json(url)
            post_data    = data[0]["data"]["children"][0]["data"]
            comment_list = data[1]["data"]["children"]

            # post title + body as its own chunk
            post_text = clean_reddit_text(
                f"{post_data.get('title', '')}\n{post_data.get('selftext', '')}"
            )
            if post_text and len(post_text.split()) >= 10:
                chunks.append({
                    "text":        post_text,
                    "source_type": "reddit",
                    "source":      sub_name,
                    "url":         url,
                    "score":       post_data.get("score", 0),
                    "depth":       -1,
                    "date":        str(int(post_data.get("created_utc", 0))),
                    "type":        "post",
                })

            for comment_data in comment_list:
                chunk_comment_tree(
                    comment_data,
                    parent_text=None,
                    chunks=chunks,
                    subreddit=sub_name,
                    post_url=url,
                    current_depth=0,
                )

            time.sleep(3)

        except Exception as e:
            print(f"  Failed: {url} -- {e}")
            continue

    print(f"  Reddit chunks collected: {len(chunks)}")
    return chunks

# ── Website ingestion ──────────────────────────────────────────────────────────

def scrape_page_text(url):
    """Fetch and extract clean body text from a webpage using BeautifulSoup."""
    resp = requests.get(url, headers=WEBSITE_HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(class_=["post-content", "entry-content", "article-body"])
        or soup.find("body")
    )

    if not main:
        return ""

    text  = main.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l and len(l) > 20]
    return "\n".join(lines)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=token_len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def scrape_websites(sources):
    chunks = []

    for source in sources:
        print(f"  Scraping {source['name']}...")
        try:
            text = scrape_page_text(source["url"])
            if not text:
                print(f"  Empty content: {source['url']}")
                continue

            text   = clean_website_text(text)   # collapse \n before splitting
            splits = splitter.split_text(text)

            for i, split in enumerate(splits):
                chunks.append({
                    "text":        split,
                    "source_type": "website",
                    "source":      source["name"],
                    "url":         source["url"],
                    "type":        "article",
                    "chunk_index": i,
                    "chunk_total": len(splits),
                })

            time.sleep(1)

        except Exception as e:
            print(f"  Failed: {source['url']} -- {e}")
            continue

    print(f"  Website chunks collected: {len(chunks)}")
    return chunks

# ── Sanity check ───────────────────────────────────────────────────────────────

def sanity_check(chunks, n=3):
    """Print n random chunks from each source type for visual inspection."""
    print("\n" + "=" * 60)
    print("SANITY CHECK -- 3 random chunks per source")
    print("=" * 60)

    for source_type in ("reddit", "website"):
        pool   = [c for c in chunks if c["source_type"] == source_type]
        sample = random.sample(pool, min(n, len(pool)))

        print(f"\n-- {source_type.upper()} (total: {len(pool)} chunks) --")
        for i, chunk in enumerate(sample, 1):
            print(f"\n  [{i}] source : {chunk['source']}")
            print(f"      type   : {chunk['type']}")
            if source_type == "reddit":
                print(f"      score  : {chunk['score']}  |  depth: {chunk['depth']}")
            print(f"      tokens : {token_len(chunk['text'])}")
            print(f"      text preview:")
            print(f"        {chunk['text'][:300]}...")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    all_chunks = []

    print("\n[ Reddit ingestion -- direct .json ]")
    reddit_chunks = scrape_reddit(REDDIT_POSTS)
    all_chunks.extend(reddit_chunks)

    print("\n[ Website ingestion -- requests + BeautifulSoup ]")
    web_chunks = scrape_websites(WEBSITE_SOURCES)
    all_chunks.extend(web_chunks)

    reddit_total  = sum(1 for c in all_chunks if c["source_type"] == "reddit")
    website_total = sum(1 for c in all_chunks if c["source_type"] == "website")
    print(f"\nTotal chunks : {len(all_chunks)}")
    print(f"  Reddit     : {reddit_total}")
    print(f"  Website    : {website_total}")

    os.makedirs("data", exist_ok=True)
    output_path = "data/skincare_chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")

    sanity_check(all_chunks)


if __name__ == "__main__":
    main()