
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings


BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENT_DIR = BASE_DIR / "data" / "documents"
VECTOR_DB_DIR = BASE_DIR / "data" / "chroma"


def load_documents(docs_dir=DOCUMENT_DIR):
    """Load supported text and Markdown files from a directory."""
    docs_dir = Path(docs_dir)

    if not docs_dir.exists():
        raise FileNotFoundError(f"Document directory not found: {docs_dir}")

    documents = []

    for file_path in sorted(docs_dir.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in {".txt", ".md"}:
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())

    return documents


def ingest_documents(docs_dir=DOCUMENT_DIR, persist_dir=VECTOR_DB_DIR, embeddings=None):
    """Load documents, split them into chunks, and create a Chroma vector store."""
    documents = load_documents(docs_dir)

    if not documents:
        raise ValueError(
            f"No supported text or Markdown documents found in {docs_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_documents(documents)

    if embeddings is None:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")

    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
    )

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")

    return vector_store


def create_vector_store():
    """Maintain compatibility with the existing ingestion script."""
    return ingest_documents()


if __name__ == "__main__":
    create_vector_store()
