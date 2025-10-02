#!/usr/bin/env python3
import os 
import time 
import hashlib 
import re 
import json 
import google.generativeai as genai 
from ragflow_sdk import RAGFlow


# === CONFIG ===
RAGFLOW_API_KEY = ""
RAGFLOW_BASE_URL = ""
GOOGLE_API_KEY = ""

# Init Gemini
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")

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

def gen_page_keywords(page_text, retries=2, pause=3):
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
    for attempt in range(1, retries + 1):
        try:
            resp = gemini_model.generate_content(prompt)
            cleaned = clean_keywords(resp.text)
            if cleaned:
                return cleaned
        except Exception as e:
            print(f"⚠️ gen_page_keywords failed on attempt {attempt}: {e}")
        time.sleep(pause*attempt)
    return []


def gen_chunk_keywords(page_text, chunk, page_keywords, retries=2, pause=3):
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
    for attempt in range(1, retries + 1):
        try:
            resp = gemini_model.generate_content(prompt)
            chunk_keywords = clean_keywords(resp.text)

            # Merge with page-level keywords only
            combined = chunk_keywords + page_keywords

            # Normalize + dedupe final list
            final_keywords = clean_keywords("\n".join(combined))

            if final_keywords:
                return final_keywords
        except Exception as e:
            print(f"⚠️ gen_chunk_keywords failed on attempt {attempt}: {e}")
        time.sleep(pause*attempt)
    return []


def gen_questions_and_answers(page_text, chunk_text, retries=2, pause=3, debug_dir="qa_debug"):
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
        try:
            resp = gemini_model.generate_content(prompt)
            raw = resp.text.strip()

            # Save raw attempt to file
            debug_file = os.path.join(debug_dir, f"qa_attempt_{attempt}.txt")
            with open(debug_file, "w") as f:
                f.write(raw)
            print(f"⚠️ Saved raw output of attempt {attempt} to {debug_file}")

            if raw.startswith("```"):
                raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
                raw = re.sub(r"\n```$", "", raw)

            qa_list = json.loads(raw)
            if isinstance(qa_list, list) and all("question" in qa and "answer" in qa for qa in qa_list):
                return qa_list
        except Exception as e:
            print(f"⚠️ gen_questions_and_answers failed on attempt {attempt}: {e}")
        time.sleep(pause*attempt)

    return []

def md5_bytes(data: bytes) -> str:
    h = hashlib.md5()
    h.update(data)
    return h.hexdigest()

def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def wait_for_chunks(doc, timeout=300):
    """Poll until Ragflow finishes parsing and chunks are available."""
    waited = 0
    while waited < timeout:
        try:
            chunks = doc.list_chunks(page=1, page_size=1)
            if chunks:
                return True
        except Exception:
            pass
        time.sleep(POLL_INTERVAL)
        waited += POLL_INTERVAL
    return False


# --- MAIN CONFIG ---
DATASET_NAME = "Alliance-Wiki-Api"
LOCAL_DIR = os.path.expanduser("~/rag-flow/alliance-wiki-rag-data/wiki_pages")

# Subdirectories for keywords + prompt completions
KEYWORDS_DIR = os.path.join(LOCAL_DIR, "keywords")
PROMPTS_DIR = os.path.join(LOCAL_DIR, "prompt_completion")

# Ensure directories exist
os.makedirs(KEYWORDS_DIR, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)

# Runtime controls
POLL_INTERVAL = 5
DEBUG = False  # set False to actually upload/parse

# === INIT ===
print("[*] Connecting to RAGFlow...")
rag = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_BASE_URL)
dataset = rag.list_datasets(name=DATASET_NAME)[0]
print(f"[✓] Using dataset: {dataset.name} ({dataset.id})")

remote_docs = dataset.list_documents(page=1, page_size=5000)
remote_map = {os.path.basename(doc.name): doc for doc in remote_docs}

for fname in sorted(os.listdir(LOCAL_DIR)):
    if not fname.endswith(".txt"):
        continue

    path = os.path.join(LOCAL_DIR, fname)

    # --- Skip redirects ---
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if len(lines) == 1 and lines[0].strip().upper().startswith("#REDIRECT"):
            print(f"⏭️ Skipping redirect: {fname}")
            continue
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        continue

    base = os.path.splitext(fname)[0]
    page_kw_file = os.path.join(KEYWORDS_DIR, f"{base}-page-keywords.txt")

    # --- Find or upload doc ---
    doc = remote_map.get(fname)
    need_upload = False

    if not doc:
        print(f"❌ Missing in Ragflow: {fname}")
        need_upload = True
    else:
        try:
            remote_bytes = doc.download()
            remote_md5 = md5_bytes(remote_bytes)
            local_md5 = md5_file(path)
            if local_md5 != remote_md5:
                print(f"⚠️ Mismatch: {fname}")
                print(f"   Local:  {local_md5}")
                print(f"   Remote: {remote_md5}")
                need_upload = True
            else:
                print(f"✅ Match: {fname}")
        except Exception as e:
            print(f"❌ Error comparing MD5 for {fname}: {e}")
            continue

    if need_upload and not DEBUG:
        with open(path, "rb") as f:
            dataset.upload_documents([{"display_name": fname, "blob": f.read()}])
        # refresh doc object
        doc = dataset.list_documents(name=fname)[0]
        dataset.async_parse_documents([doc.id])

        # Wait until chunks appear
        print(f"⏳ Waiting for chunks for {fname} ...")
        for attempt in range(60):  # wait up to 5 min
            chunks = doc.list_chunks(page=1, page_size=1)
            if chunks:
                print(f"✅ Chunks ready for {fname}")
                break
            time.sleep(POLL_INTERVAL)
        else:
            print(f"❌ Timeout waiting for chunks: {fname}")
            continue

    # --- Page keywords ---
    if os.path.exists(page_kw_file):
        with open(page_kw_file, "r") as f:
            page_keywords = [line.strip() for line in f if line.strip()]
        print(f"🔑 Using existing page keywords ({len(page_keywords)})")
    else:
        if DEBUG:
            print(f"   [DEBUG] Would generate page keywords for {fname}")
            page_keywords = []
        else:
            page_text = doc.download().decode("utf-8", errors="ignore")
            page_keywords = gen_page_keywords(page_text)
            with open(page_kw_file, "w") as f:
                f.write("\n".join(page_keywords))
            print(f"[✓] Saved page keywords for {fname}")

    # --- Gather all chunks ---
    chunks = []
    for cpage in range(1, 20):
        page_chunks = doc.list_chunks(page=cpage, page_size=50)
        if not page_chunks:
            break
        chunks.extend(page_chunks)
    print(f"📑 Found {len(chunks)} chunks for {fname}")

    # --- Process chunks ---
    page_text = doc.download().decode("utf-8", errors="ignore")
    for i, chunk in enumerate(chunks, start=1):
        kw_file = os.path.join(KEYWORDS_DIR, f"{base}-{i}-keywords.txt")
        qa_file = os.path.join(PROMPTS_DIR, f"{base}-{i}-qa.json")

        kws = None
        need_qna = False

        # Keywords
        if os.path.exists(kw_file):
            with open(kw_file, "r") as f:
                kws = [line.strip() for line in f if line.strip()]
            print(f"   🔑 Chunk {i}: using existing {len(kws)} keywords")
            if not DEBUG:
                chunk.update({"important_keywords": kws})
        else:
            if DEBUG:
                print(f"   [DEBUG] Would generate keywords for chunk {i}")
            else:
                kws = gen_chunk_keywords(page_text, chunk, page_keywords)
                with open(kw_file, "w") as f:
                    f.write("\n".join(kws))
                chunk.update({"important_keywords": kws})
                print(f"[✓] Chunk {i}: generated {len(kws)} keywords")
            need_qna = True  # if keywords were just generated, also gen Q&A

        # Q&A
        if os.path.exists(qa_file) and not need_qna:
            print(f"   📝 Chunk {i}: Q&A already exists")
        else:
            if DEBUG:
                print(f"   [DEBUG] Would generate Q&A for chunk {i}")
            else:
                qa_pairs = gen_questions_and_answers(page_text, chunk.content)
                if qa_pairs:
                    with open(qa_file, "w") as f:
                        json.dump(qa_pairs, f, indent=2)
                    print(f"[✓] Chunk {i}: saved {len(qa_pairs)} Q&A pairs")
                else:
                    print(f"❌ Chunk {i}: failed to generate Q&A")


    # 🔽 Add this at the end of the per-file loop
    time.sleep(3)
