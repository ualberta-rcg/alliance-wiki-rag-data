from ragflow_sdk import RAGFlow
import hashlib
import os

# === CONFIG ===
RAGFLOW_API_KEY = ""
RAGFLOW_BASE_URL = ""
DATASET_NAME = "Alliance-Wiki-Api"
LOCAL_DIR = os.path.expanduser(
    "~/rag-flow/alliance-wiki-rag-data/wiki_pages"
)

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

try:
    docs = dataset.list_documents(page=1, page_size=1000)
except Exception as e:
    print(f"❌ Error listing documents: {e}")
    exit(1)

# Reverse the list and take first 100
docs = list(reversed(docs))

# Keep track of mismatches
mismatches = []

for i, doc in enumerate(docs, start=1):
    print(f"\n=== Document {i} ===")
    try:
        print(f"RAGFlow: {doc.name} (id={doc.id})")

        # Compute RAGFlow hash
        remote_bytes = doc.download()
        remote_md5 = md5_bytes(remote_bytes)
        print(f"  Remote MD5: {remote_md5}")

        # Match to local file (strip numeric prefix if present)
        base_name = doc.name.split("/", 1)[-1]
        local_path = os.path.join(LOCAL_DIR, base_name)

        if os.path.exists(local_path):
            try:
                local_md5 = md5_file(local_path)
                print(f"  Local: {local_path}")
                print(f"  Local MD5:  {local_md5}")

                if remote_md5 == local_md5:
                    print("  ✅ Match")
                else:
                    print("  ❌ Mismatch")
                    mismatches.append((local_path, local_md5, remote_md5))

            except Exception as e:
                print(f"  ❌ Error computing MD5 for local file {local_path}: {e}")
        else:
            print(f"  Local file not found: {local_path}")

    except Exception as e:
        print(f"❌ Error processing document {doc.name}: {e}")

# --- Summary ---
if mismatches:
    print("\n=== Mismatched Files ===")
    for path, local_md5, remote_md5 in mismatches:
        print(f"{path}")
        print(f"   Local:  {local_md5}")
        print(f"   Remote: {remote_md5}")
else:
    print("\n✅ All local files match their RAGFlow versions")

