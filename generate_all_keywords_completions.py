import google.generativeai as genai
from ragflow_sdk import RAGFlow
import os
import re
import json

# === CONFIG ===
RAGFLOW_API_KEY = os.getenv("RAGFLOW_API_KEY")
RAGFLOW_BASE_URL = os.getenv("RAGFLOW_BASE_URL")
DATASET_NAME = os.getenv("RAGFLOW_DATASET_NAME")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Init Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Init RAGFlow
rag = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_BASE_URL)

# --- UTILITIES ---

def clean_keywords(text, limit=None):
    lines = re.split(r'[\n,;]', text)
    keywords, seen, clean_list = [], set(), []
    for line in lines:
        line = re.sub(r"^[\d\-•.\)\( ]+", "", line).strip().lower()
        line = re.sub(r"[^a-z0-9 _-]", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line and not any(char.isdigit() for char in line):
            keywords.append(line)
    for k in keywords:
        if k not in seen:
            seen.add(k)
            clean_list.append(k)
    return clean_list[:limit] if limit else clean_list

def gen_page_keywords(page_text):
    prompt = f"""
From the following document, extract 6–10 short, high-level keywords.

Formatting rules:
- Output only a plain list (one keyword or phrase per line).
- Use only lowercase letters, numbers, hyphens, or spaces.
- Do not include numbering, bullets, or special characters.
- Avoid generic terms like 'documentation', 'page', 'MediaWiki'.

Document:
{page_text}
"""
    resp = gemini_model.generate_content(prompt)
    return clean_keywords(resp.text)


def gen_page_keywords(page_text):
    prompt = f"""
From the following document, extract 6–10 short, high-level keywords.

Formatting rules:
- Output only a plain list (one keyword or phrase per line).
- Use only lowercase letters, numbers, hyphens, or spaces.
- Do not include numbering, bullets, or special characters.
- Avoid generic terms like 'documentation', 'page', 'MediaWiki'.

Document:
{page_text}
"""
    resp = gemini_model.generate_content(prompt)
    return clean_keywords(resp.text)


def gen_chunk_keywords(page_text, chunk, page_keywords):
    """
    Generate chunk-specific keywords.
    - Pass existing important_keywords as hints in the prompt (not merged).
    - Always append page-level keywords at the end.
    - Normalize and deduplicate before returning.
    """
    chunk_text = chunk.content
    existing_keywords = getattr(chunk, "important_keywords", []) or []

    prompt = f"""
You are tagging a section of a technical document with relevant search keywords.

Instructions:
- First consider the **whole page** (context).
- Then analyze the **specific chunk** (focus).
- If helpful, consider these suggested keywords: {", ".join(existing_keywords) if existing_keywords else "none"}.
- Output 8–12 keywords or phrases, one per line about the **specific chunk**.
- Use only lowercase letters, numbers, hyphens, or spaces.
- Do not include bullets, numbers, or special characters.
- Avoid generic or structural terms like 'page', 'view', 'content'.
- Provide no other text other than the list.

Page (context):
{page_text}

Chunk (focus):
{chunk_text}
"""
    resp = gemini_model.generate_content(prompt)
    chunk_keywords = clean_keywords(resp.text)

    # Merge with page-level keywords only
    combined = chunk_keywords + page_keywords

    # Normalize + dedupe final list
    final_keywords = clean_keywords("\n".join(combined))

    return final_keywords




def gen_questions_and_answers(page_text, chunk_text, retries=2, debug_dir="qa_debug"):
    os.makedirs(debug_dir, exist_ok=True)

    prompt = f"""
You are generating **question–answer pairs** from a technical document.

Requirements:
- Output only valid JSON.
- JSON must be a list of objects, each with keys "question" and "answer".
- "question" must be a single string (natural user question).
- "answer" must be a single string that can be answered from the chunk (with page context if needed).
- Do not include any text outside of the JSON block.
- Create as many questions and answers you can with the given information.


Example:
[
  {{"question": "How do you install ZFS on Ubuntu?", "answer": "Run 'sudo apt-get install zfsutils-linux' after updating packages."}},
  {{"question": "How do you create a zpool in ZFS?", "answer": "Use 'sudo zpool create -f data /dev/vdb /dev/vdc' to create a new pool."}}
]

Document (context):
{page_text}

Chunk (focus):
{chunk_text}
"""
    for attempt in range(1, retries + 1):
        resp = gemini_model.generate_content(prompt)
        raw = resp.text.strip()

        # Save raw attempt to file
        debug_file = os.path.join(debug_dir, f"qa_attempt_{attempt}.txt")
        with open(debug_file, "w") as f:
            f.write(raw)
        print(f"⚠️ Saved raw output of attempt {attempt} to {debug_file}")

        try:
            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
                raw = re.sub(r"\n```$", "", raw)

            qa_list = json.loads(raw)
            if isinstance(qa_list, list) and all("question" in qa and "answer" in qa for qa in qa_list):
                return qa_list
        except Exception as e:
            print(f"⚠️ Invalid JSON on attempt {attempt}: {e}")

    return []

# --- MAIN ---

keywords_dir = "keywords"
prompts_dir = "prompt_completion"
os.makedirs(keywords_dir, exist_ok=True)
os.makedirs(prompts_dir, exist_ok=True)

dataset = rag.list_datasets(name=DATASET_NAME)[0]
page = 1

while True:
    docs = dataset.list_documents(page=page, page_size=10)
    if not docs:
        break

    for doc in docs:
        print(f"\n=== Document: {doc.name} ===")
        base_name = os.path.splitext(os.path.basename(doc.name))[0]

        # Download full text
        page_text = doc.download().decode("utf-8", errors="ignore")

        # Page-level keywords
        page_keywords = gen_page_keywords(page_text)
        with open(os.path.join(keywords_dir, f"{base_name}-page-keywords.txt"), "w") as f:
            f.write("\n".join(page_keywords))
        print(f"Saved {base_name}-page-keywords.txt")

        # Gather chunks
        chunks = []
        for cpage in range(1, 20):  # adjust if needed
            page_chunks = doc.list_chunks(page=cpage, page_size=50)
            if not page_chunks:
                break
            chunks.extend(page_chunks)
        print(f"Found {len(chunks)} chunks")

        # Process chunks
        for i, chunk in enumerate(chunks, start=1):
            print(f"\n--- Chunk {i} ---")
            print(chunk.content[:200], "...\n")

            # Keywords
            try:
                kws = gen_chunk_keywords(page_text, chunk, page_keywords)
                with open(os.path.join(keywords_dir, f"{base_name}-{i}-keywords.txt"), "w") as f:
                    f.write("\n".join(kws))
                print(f"Saved {base_name}-{i}-keywords.txt")
            except Exception as e:
                print(f"❌ Keyword generation failed: {e}")

            # Q&A
            try:
                qa_pairs = gen_questions_and_answers(page_text, chunk.content)
                if qa_pairs:
                    with open(os.path.join(prompts_dir, f"{base_name}-{i}-qa.json"), "w") as f:
                        json.dump(qa_pairs, f, indent=2)
                    print(f"Saved {base_name}-{i}-qa.json ({len(qa_pairs)} Q&A pairs)")
                else:
                    print(f"❌ No valid Q&A for chunk {i}")
            except Exception as e:
                print(f"❌ Q&A generation failed: {e}")

    page += 1
