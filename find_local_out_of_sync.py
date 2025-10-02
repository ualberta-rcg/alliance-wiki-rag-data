from ragflow_sdk import RAGFlow
import hashlib
import os

# === CONFIG ===
RAGFLOW_API_KEY = ""
RAGFLOW_BASE_URL = ""
DATASET_NAME = ""
LOCAL_DIR = os.path.expanduser("~/rag-flow/alliance-wiki-rag-data/wiki_pages")

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

# === INIT ===
rag = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_BASE_URL)

try:
    dataset = rag.list_datasets(name=DATASET_NAME)[0]
except Exception as e:
    print(f"❌ Error fetching dataset '{DATASET_NAME}': {e}")
    exit(1)

# get all remote docs (expand page_size if needed)
remote_docs = dataset.list_documents(page=1, page_size=5000)
remote_map = {os.path.basename(doc.name): doc for doc in remote_docs}

missing, skipped, matched, mismatched = [], [], [], []

for fname in sorted(os.listdir(LOCAL_DIR)):
    if not fname.endswith(".txt"):
        continue

    path = os.path.join(LOCAL_DIR, fname)

    # check if it's a redirect-only file
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if len(lines) == 1 and lines[0].strip().upper().startswith("#REDIRECT"):
            print(f"⏭️ Skipping redirect: {fname}")
            skipped.append(fname)
            continue
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")
        continue

    if fname not in remote_map:
        print(f"❌ Missing in Ragflow: {fname}")
        missing.append(fname)
        continue

    # MD5 compare
    doc = remote_map[fname]
    try:
        remote_bytes = doc.download()
        remote_md5 = md5_bytes(remote_bytes)
    except Exception as e:
        print(f"❌ Error downloading remote {fname}: {e}")
        continue

    try:
        local_md5 = md5_file(path)
    except Exception as e:
        print(f"❌ Error computing MD5 for {fname}: {e}")
        continue

    if local_md5 == remote_md5:
        print(f"✅ Match: {fname}")
        matched.append(fname)
    else:
        print(f"⚠️ Mismatch: {fname}")
        print(f"   Local:  {local_md5}")
        print(f"   Remote: {remote_md5}")
        mismatched.append(fname)

# --- Summary ---
print("\n=== Summary ===")
print(f"Skipped (redirects): {len(skipped)}")
print(f"Missing in Ragflow: {len(missing)}")
print(f"Matched:            {len(matched)}")
print(f"Mismatched:         {len(mismatched)}\n")

if missing:
    print("❌ Missing files:")
    for m in missing:
        print("  -", m)

if mismatched:
    print("\n⚠️ Mismatched files:")
    for m in mismatched:
        print("  -", m)

