#!/usr/bin/env python3
import os 
import time 
import hashlib 
import re 
import json 
import google.generativeai as genai 
from ragflow_sdk import RAGFlow
from google.api_core.exceptions import ResourceExhausted  # or generic Exception


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


def gen_questions_and_answers(page_text, chunk_text,
                              retries=3, pause=3, debug_dir="qa_debug"):
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

Document (context):
{page_text}

Chunk (focus):
{chunk_text}
"""

    for attempt in range(1, retries + 1):
        try:
            resp = gemini_model.generate_content(prompt)
            raw = resp.text.strip()

            # save raw attempt
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
            # Handle quota errors specially
            if hasattr(e, "errors") and e.errors and "quota" in str(e.errors[0]).lower():
                # Google APIs usually give retry_delay.seconds
                retry_sec = getattr(getattr(e, "retry_delay", None), "seconds", None)
                if retry_sec is None:
                    retry_sec = pause * attempt * 2  # fallback
                print(f"⚠️ Quota exceeded. Sleeping {retry_sec}s before retry…")
                time.sleep(retry_sec)
            else:
                print(f"⚠️ gen_questions_and_answers failed on attempt {attempt}: {e}")
                time.sleep(pause * attempt)

    return []


def gen_questions_and_answers_old(page_text, chunk_text, retries=2, pause=3, debug_dir="qa_debug"):
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
LOCAL_DIR = os.path.expanduser("/opt/helpy-cron/alliance-wiki-rag-data/wiki_pages")

# Subdirectories for keywords + prompt completions
KEYWORDS_DIR = os.path.join(LOCAL_DIR, "keywords")
PROMPTS_DIR = os.path.join(LOCAL_DIR, "prompt_completion")

# Ensure directories exist
os.makedirs(KEYWORDS_DIR, exist_ok=True)
os.makedirs(PROMPTS_DIR, exist_ok=True)

# Runtime controls
POLL_INTERVAL = 5
DEBUG = False
# set False to actually upload/parse

# === INIT ===
print("[*] Connecting to RAGFlow...")
rag = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_BASE_URL)
dataset = rag.list_datasets(name=DATASET_NAME)[0]
print(f"[✓] Using dataset: {dataset.name} ({dataset.id})")

remote_docs = dataset.list_documents(page=1, page_size=5000)
remote_map = {os.path.basename(doc.name): doc for doc in remote_docs}
# === MAIN ===

# Iterate local .txt pages
for fname in sorted(os.listdir(LOCAL_DIR)):
    if not fname.endswith(".txt"):
        continue

    path = os.path.join(LOCAL_DIR, fname)

    # Skip MediaWiki redirects
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if len(lines) == 1 and lines[0].strip().upper().startswith("#REDIRECT"):
            print(f"⏭️  Skipping redirect: {fname}")
            continue
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        continue

    base = os.path.splitext(fname)[0]
    page_kw_file = os.path.join(KEYWORDS_DIR, f"{base}-page-keywords.txt")

    if fname.startswith("⧼") and fname.endswith("⧽.txt"):
        print(f"⏭️  Skipping UI message page: {fname}")
        continue

    # Find remote doc (if any)
    doc = remote_map.get(fname)
    need_upload = False
    force_regen_artifacts = False

    # Compute local MD5 once
    try:
        local_md5 = md5_file(path)
    except Exception as e:
        print(f"❌ MD5 failed for local file {fname}: {e}")
        continue

    if not doc:
        print(f"❌ Missing in Ragflow: {fname}")
        need_upload = True
        force_regen_artifacts = True

        if DEBUG:
            print(f"   [DEBUG] Would upload & parse {fname}")
            # Skip further processing, because doc is None
            time.sleep(1)
            continue

    else:
        # Compare MD5s
        try:
            remote_bytes = doc.download()

            if isinstance(remote_bytes, str):
                remote_bytes = remote_bytes.encode("utf-8", errors="ignore")
            elif isinstance(remote_bytes, list) or isinstance(remote_bytes, dict):
                remote_bytes = json.dumps(remote_bytes, ensure_ascii=False).encode("utf-8", errors="ignore")

            remote_md5 = md5_bytes(remote_bytes)
        except Exception as e:
            print(f"❌ Error downloading remote {fname}: {e}")
            # If we can’t compare, treat as needing reupload
            remote_md5 = None

        if remote_md5 is None or local_md5 != remote_md5:
            print(f"⚠️  MD5 mismatch/newer detected for {fname}")
            print(f"   Local : {local_md5}")
            print(f"   Remote: {remote_md5 or 'unavailable'}")
            if DEBUG:
                print(f"   [DEBUG] Would delete & reupload {fname}")
                # ⬇️ ADD THIS
                time.sleep(1)
                continue
            # Delete old remote and reupload
            if not DEBUG:
                try:
                    # Prefer a doc-level delete if available
                    try:
                        doc.delete()
                    except AttributeError:
                        dataset.delete_documents([doc.id])
                    print(f"🗑️  Deleted old document in Ragflow: {fname}")
                except Exception as e:
                    print(f"❌ Failed to delete {fname} in Ragflow: {e}")
                    # Best effort: proceed anyway
                need_upload = True
                force_regen_artifacts = True
            else:
                print(f"   [DEBUG] Would delete & reupload {fname}")
        else:
            print(f"✅ Match: {fname}")

    # Upload & parse if needed
    if need_upload and not DEBUG:
        try:
            with open(path, "rb") as f:
                dataset.upload_documents([{"display_name": fname, "blob": f.read()}])
            # Refresh doc handle
            doc = dataset.list_documents(name=fname)[0]
            dataset.async_parse_documents([doc.id])
            print(f"⏳ Parsing {fname} ...")
            if wait_for_chunks(doc, timeout=300):
                print(f"✅ Chunks ready for {fname}")
            else:
                print(f"❌ Timeout waiting for chunks: {fname}")
                time.sleep(3)
                continue
        except Exception as e:
            print(f"❌ Upload/parse failed for {fname}: {e}")
            time.sleep(3)
            continue

    # Ensure page-level keywords
    if force_regen_artifacts or not os.path.exists(page_kw_file):
        if DEBUG:
            print(f"   [DEBUG] Would (re)generate page keywords for {fname}")
            page_keywords = []
        else:
            try:
                page_text = doc.download().decode("utf-8", errors="ignore")
                page_keywords = gen_page_keywords(page_text)
                os.makedirs(KEYWORDS_DIR, exist_ok=True)
                with open(page_kw_file, "w") as f:
                    f.write("\n".join(page_keywords))
                print(f"[✓] Saved page keywords for {fname} ({len(page_keywords)})")
            except Exception as e:
                print(f"❌ Failed generating page keywords for {fname}: {e}")
                page_keywords = []
    else:
        with open(page_kw_file, "r") as f:
            page_keywords = [line.strip() for line in f if line.strip()]
        print(f"🔑 Using existing page keywords for {fname} ({len(page_keywords)})")

    if doc is None:
        print(f"⚠️  No doc handle after preflight, skipping: {fname}")
        continue

    chunks = []
    for cpage in range(1, 100):  # generous cap
        page_chunks = doc.list_chunks(page=cpage, page_size=50)
        if not page_chunks:
            break
        chunks.extend(page_chunks)
    print(f"📑 Found {len(chunks)} chunks for {fname}")

    # Full page text (for prompts)
    try:
        page_text = doc.download().decode("utf-8", errors="ignore")
    except Exception:
        page_text = ""

    # For each chunk: ensure keyword file + Q&A
    for i, chunk in enumerate(chunks, start=1):
        kw_file = os.path.join(KEYWORDS_DIR, f"{base}-{i}-keywords.txt")
        qa_file = os.path.join(PROMPTS_DIR, f"{base}-{i}-qa.json")

        # Keywords for chunk
        regen_kw = force_regen_artifacts or (not os.path.exists(kw_file) or os.path.getsize(kw_file) == 0)
        if regen_kw:
            if DEBUG:
                print(f"   [DEBUG] Would generate keywords for chunk {i}")
                kws = []
            else:
                try:
                    kws = gen_chunk_keywords(page_text, chunk, page_keywords)
                    os.makedirs(KEYWORDS_DIR, exist_ok=True)
                    with open(kw_file, "w") as f:
                        f.write("\n".join(kws))
                    print(f"[✓] Chunk {i}: generated {len(kws)} keywords")
                except Exception as e:
                    print(f"❌ Chunk {i}: keyword generation failed: {e}")
                    kws = []
        else:
            with open(kw_file, "r") as f:
                kws = [line.strip() for line in f if line.strip()]
            print(f"   🔑 Chunk {i}: using existing {len(kws)} keywords")

        # Push keywords back to Ragflow chunk (best-effort)
        if not DEBUG:
            try:
                if kws:
                   chunk.update({"important_keywords": kws})
            except Exception as e:
                print(f"⚠️  Chunk {i}: failed to update important_keywords: {e}")

        # Q&A for chunk
        regen_qa = force_regen_artifacts or (not os.path.exists(qa_file) or os.path.getsize(qa_file) == 0)
        if regen_qa:
            if DEBUG:
                print(f"   [DEBUG] Would generate Q&A for chunk {i}")
            else:
                try:
                    qa_pairs = gen_questions_and_answers(page_text, chunk.content)
                    if qa_pairs:
                        os.makedirs(PROMPTS_DIR, exist_ok=True)
                        with open(qa_file, "w") as f:
                            json.dump(qa_pairs, f, indent=2)
                        print(f"[✓] Chunk {i}: saved {len(qa_pairs)} Q&A pairs")
                    else:
                        print(f"❌ Chunk {i}: failed to generate Q&A")
                except Exception as e:
                    print(f"❌ Chunk {i}: Q&A generation error: {e}")
        else:
            print(f"   📝 Chunk {i}: Q&A already exists")

    # polite pacing between files only if we actually hit GenAI API
    if not DEBUG and (need_upload or force_regen_artifacts):
        time.sleep(3)
