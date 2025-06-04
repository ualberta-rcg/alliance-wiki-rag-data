from ragflow_sdk import RAGFlow
from openai import OpenAI
import re

import argparse

parser = argparse.ArgumentParser(description="Ask a question to RAG-powered assistant.")
parser.add_argument("question", type=str, help="The question to ask", nargs="+")
args = parser.parse_args()

# Join the question back into a string (in case it's passed as multiple words)
query = " ".join(args.question)

# === CONFIG ===
RAGFLOW_API_KEY = ""
RAGFLOW_BASE_URL = "http://34.118.144.23"
OPENAI_API_KEY = ""
DATASET_NAME = "Alliance-Wiki-HTML"

# === INIT ===
rag = RAGFlow(api_key=RAGFLOW_API_KEY, base_url=RAGFLOW_BASE_URL)
client = OpenAI(api_key=OPENAI_API_KEY)

# === RETRIEVE DATASET ===
dataset = rag.list_datasets(name=DATASET_NAME)[0]

# === QUERY ===
#query = "How would a user get access to Vulcan or TamAI clusters. Are there any special steps? ( tamia, vulcan, cluster )"

print(f"Received question: {query}")


# === RUN RETRIEVAL ===
results = rag.retrieve(
    question=query,
    dataset_ids=[dataset.id],
    page=1,
    page_size=15,
    similarity_threshold=0.15,
    vector_similarity_weight=0.25,
    top_k=2048
)

# === COMBINE CONTENT ===
combined_chunks = "\n\n".join([chunk.content.strip() for chunk in results])
context_prompt = f"""You are a helpful assistant. Answer the user's question using only the information provided below. Be detailed and cite clusters properly.

Context:
{combined_chunks}

Question: {query}
"""

# === OPENAI CALL ===
response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.8,
    messages=[
        {"role": "system", "content": "You are a highly intelligent assistant that answers using only the given context."},
        {"role": "user", "content": context_prompt}
    ]
)

# === PRINT ANSWER ===
print("\n=== ANSWER ===\n")
print(response.choices[0].message.content)

