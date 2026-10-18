from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


BASE_DIR = Path(__file__).resolve().parent.parent
VECTOR_DB_DIR = str(BASE_DIR / "data" / "chroma")


class RAGPipeline:

    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        self.vector_store = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embeddings
        )

        self.llm = ChatOllama(
            model="llama3.2",
            temperature=0
        )

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 3}
        )

    def ask(self, question):

        documents = self.retriever.invoke(question)

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context, say:

"I don't have enough information to answer that."

Do not make up information.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(prompt)

        return {
            "question": question,
            "answer": response.content,
            "context": [
                document.page_content
                for document in documents
            ],
            "sources": [
                document.metadata.get("source")
                for document in documents
            ]
        }
