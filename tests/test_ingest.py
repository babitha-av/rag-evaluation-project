
import hashlib

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from app.ingest import ingest_documents, load_documents


class FakeEmbeddings(Embeddings):
    """Deterministic local embeddings for unit tests."""

    def _embed(self, text):
        vector = [0.0] * 128

        for word in text.lower().split():
            index = int(
                hashlib.md5(word.encode("utf-8")).hexdigest(), 16
            ) % len(vector)
            vector[index] += 1.0

        return vector

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)


def test_load_documents_reads_text_files(tmp_path):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    (docs_dir / "alpha.txt").write_text(
        "Alpha content for retrieval.", encoding="utf-8"
    )
    (docs_dir / "beta.md").write_text(
        "# Beta\n\nThis is beta content.", encoding="utf-8"
    )

    documents = load_documents(docs_dir)

    assert len(documents) == 2
    assert all(isinstance(doc, Document) for doc in documents)
    assert any("Alpha content" in doc.page_content for doc in documents)
    assert any("beta content" in doc.page_content.lower() for doc in documents)


def test_ingest_documents_creates_vectorstore(tmp_path):
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()

    (docs_dir / "sample.txt").write_text(
        "Acme Health offers telehealth and wellness plans.",
        encoding="utf-8",
    )

    vectorstore_dir = tmp_path / "vectorstore"

    vectorstore = ingest_documents(
        docs_dir,
        persist_dir=vectorstore_dir,
        embeddings=FakeEmbeddings(),
    )

    results = vectorstore.similarity_search("telehealth", k=1)

    assert len(results) == 1
    assert "telehealth" in results[0].page_content.lower()
