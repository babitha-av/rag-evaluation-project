
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

    def _expand_query(self, question):
        """Add known domain synonyms to improve retrieval."""
        normalized = question.lower()

        if "mfa" in normalized:
            return (
                f"{question} multi-factor authentication "
                "account security settings enable"
            )

        if "multi-factor authentication" in normalized:
            return (
                f"{question} MFA account security settings enable"
            )

        return question

    def ask(self, question):

        # Improve retrieval for known terminology variants.
        retrieval_query = self._expand_query(question)
        documents = self.retriever.invoke(retrieval_query)

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
You are a helpful question-answering assistant for AcmeAI.

Use ONLY the provided context to answer factual questions.
Do not invent facts or assume that missing information is true.

POLICY REASONING:
- Identify applicable eligibility rules, deadlines, and limits.
- Compare the user's stated situation with those conditions.
- Apply the stated conditions before deciding eligibility.
- Do not confuse a policy's eligibility window with a general
  processing time.
ANSWERING AND ABSTENTION:
- First, check whether the provided context contains the facts
  needed to answer the question.
- If the context directly answers the question, answer clearly
  using those facts.
- If the question describes a scenario, apply the relevant
  conditions and limits in the context to that scenario.
- Do not abstain merely because the answer requires a simple
  comparison or inference from facts explicitly stated in the context.
- Abstain only when the information needed to answer is genuinely
  absent from the context.
- Never invent facts that are not supported by the context.

AMBIGUITY:
- If the user asks for a price but does not specify which plan,
  and multiple plans are available in the context, ask which
  plan they mean instead of listing all plan prices.
- Ask a concise clarifying question when essential information
  is missing.
- Do not ask for clarification when the context and question
  already provide enough information to answer.

Context:
{context}

Original user question:
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
