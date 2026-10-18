from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

VECTOR_DB_DIR = "data/chroma"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

db = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)

print("Number of documents:", db._collection.count())

results = db.similarity_search(
    "How long is the refund period?",
    k=3
)

print("\nRetrieved documents:")

for i, document in enumerate(results, 1):
    print(f"\n--- Document {i} ---")
    print(document.page_content)
    print("Metadata:", document.metadata)
