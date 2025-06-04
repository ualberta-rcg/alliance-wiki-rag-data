from ragflow_sdk import RAGFlow
import openai  # or use your preferred LLM

# Setup
rag = RAGFlow(api_key="", base_url="http://34.118.144.23")
dataset = rag.list_datasets(name="Alliance-Public-Wiki")[0]
doc = dataset.list_documents()[0]

# Download the document
text = doc.download().decode("utf-8")

# Generate keywords using your LLM
openai.api_key = ""
prompt = f"Extract 10 important keywords or phrases that summarize the following document:\n{text[:4000]}"
keywords_response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}]
)

doc_keywords = keywords_response.choices[0].message.content.strip().split("\n")
doc_keywords = [k.strip("- •") for k in doc_keywords if k.strip()]

chunks = doc.list_chunks()
for chunk in chunks:
    prompt = f"Extract keywords from this passage:\n\n{chunk.content}\n\nReturn a comma-separated list."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    chunk_keywords = response.choices[0].message.content.strip().split(",")
    chunk_keywords = [k.strip() for k in chunk_keywords]

    # Combine with document-level keywords
    combined_keywords = list(set(doc_keywords + chunk_keywords))

    # Update chunk with combined keywords
    chunk.update({
        "important_keywords": combined_keywords
    })


for chunk in chunks:
    prompt = f"""Generate 2 - 4 question and answer pairs based on this passage:

{chunk.content}

Format:
Q1: ...
A1: ...
Q2: ...
A2: ...
Q3: ...
A3: ...
Q4: ...
A4: ...
"""

    qa_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    qa_text = qa_response.choices[0].message.content.strip()

    # Optionally: Add these as new chunks
    doc.add_chunk(content=qa_text, important_keywords=doc_keywords)

