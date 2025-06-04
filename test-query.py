from ragflow_sdk import RAGFlow

# === CONFIG ===
RAGFLOW_API_KEY = ""
RAGFLOW_BASE_URL = "http://34.118.144.23"
DATASET_NAME = "Alliance-Wiki-HTML"

# === INIT ===
rag = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_BASE_URL)

# === LOOKUP DATASET ID ===
dataset = rag.list_datasets(name=DATASET_NAME)[0]

# === SEARCH QUERY ===
query = "Tell me about the Vulcan, TamAI, Cedar, and Kilarny clusters. As much detail as you can."

# === RUN RETRIEVAL ===
results = rag.retrieve(
    question=query,
    dataset_ids=[dataset.id],
    page=1,
    page_size=10,
    similarity_threshold=0.25,
    vector_similarity_weight=0.45,
    top_k=1024
)

# === DISPLAY RESULTS ===
for i, chunk in enumerate(results, 1):
    print(f"\n--- Result {i} ---")
    print(f"Chunk ID: {chunk.id}")
    print(f"Document ID: {chunk.document_id}")
    print(f"Keywords: {chunk.important_keywords}")
    print("Content:")
    print(chunk.content)

