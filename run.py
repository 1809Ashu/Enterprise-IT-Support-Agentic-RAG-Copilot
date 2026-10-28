from app.services.ingestion import load_file, chunk_documents
from pathlib import Path
from app.rag.vectorstores import add_documents

docs = load_file(Path("data/sample_kb/company_it_handbook.md"))
chunks = chunk_documents(docs)

add_documents(chunks)
print(f"Added {len(chunks)} chunks to the vector store.")