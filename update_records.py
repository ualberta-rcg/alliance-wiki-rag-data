from ragflow_sdk import RAGFlow
from openai import OpenAI
import re
import time

# === CONFIG ===
RAGFLOW_API_KEY = ""
RAGFLOW_BASE_URL = "http://34.118.144.23"
OPENAI_API_KEY = ""
DATASET_NAME = "Alliance-Wiki-HTML"

# === PARSER CONFIG TO APPLY ===
PARSER_CONFIG = {
    "auto_keywords": 8,
    "auto_questions": 0,
    "chunk_token_num": 1024,
    "delimiter": "\n",
    "html4excel": False,
    "layout_recognize": "DeepDOC",
    "raptor": {"use_raptor": False},
    "graphrag": {"use_graphrag": False}
}

# === INIT ===
rag = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_BASE_URL)
client = OpenAI(api_key=OPENAI_API_KEY)

def clean_keywords(text, limit):
    lines = re.split(r'[\n,;]', text)
    keywords = []
    for line in lines:
        line = line.strip()
        line = re.sub(r"^[\d\-•.\)\( ]+", "", line)
        if line and not any(char.isdigit() for char in line):
            keywords.append(line)
    return keywords[:limit]

# === MAIN ===
dataset = rag.list_datasets(name=DATASET_NAME)[0]
documents = []
for page in range(1, 100):
    page_docs = dataset.list_documents(page=page, page_size=50)
    if not page_docs:
        break
    documents.extend(page_docs)



for doc in documents:
    print(f"\nUpdating and re-parsing: {doc.name}")

    # === Update parser config and force re-chunking ===
    try:
        doc.update({
            "chunk_method": "naive",
            "parser_config": PARSER_CONFIG
        })
        dataset.async_parse_documents([doc.id])
        print("Parser config applied and parse triggered.")
    except Exception as e:
        print(f"Failed to update/parse doc: {e}")
        continue

    # === Wait for processing (or check back later if doing async batch processing) ===
    print("Sleeping 10s to let parsing start (you can increase this if needed)...")
    time.sleep(10)

    # === DOC KEYWORDS ===
    try:
        text = doc.download().decode("utf-8")
        prompt = f"""
You are extracting search keywords to help users find the content within this document more effectively.

From the following document, extract **up to 6 clean and meaningful keywords or phrases** that:
- Are specific to the topic, technology, or task described within the content
- Reflect core content or terminology
- Would be helpful in a search query
- Keywords that reflect the content, and not the platform the content is running on, or very basic terms.
- Avoid generic terms like 'documentation', 'MediaWiki', 'page', or 'view'

Document content:
{text[:4000]}

Return a comma-separated list of keywords only.
"""
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        doc_keywords = clean_keywords(resp.choices[0].message.content, 6)
    except Exception as e:
        print(f"Keyword generation failed for doc: {e}")
        doc_keywords = []

    print(f"Doc Keywords: {doc_keywords}")

    # === CHUNK KEYWORDS ===
    try:
        chunks = []
        for page in range(1, 100):
            page_chunks = doc.list_chunks(page=page, page_size=50)
            if not page_chunks:
                break
            chunks.extend(page_chunks)

        for chunk in chunks:
            prompt = f"""
You are tagging a small section of a technical document with **relevant search keywords**.

From the following passage, extract **up to 8 keywords or phrases** that:
- Reflect core content or terminology
- Keywords that reflect the content, and not the platform the content is running on, or very basic terms.
- Capture technical concepts, commands, tools, technologies, or named entities
- Would help someone search for this specific section
- Exclude generic or structural terms like 'page', 'view', 'MediaWiki', or 'content'

Passage:
{chunk.content}

Return a comma-separated list of keywords only.
"""
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            chunk_keywords = clean_keywords(resp.choices[0].message.content, 8)
            combined = list(set(doc_keywords + chunk_keywords))
            chunk.update({"important_keywords": combined})
            print(f"Updated chunk with: {combined}")
    except Exception as e:
        print(f"Chunk keyword update failed: {e}")

