"""
RAG chain:
- Loads Chroma vectorstore
- Uses retriever to get relevant chunks
- Prompts LLM with grounded context
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from .config import settings
from .llm_factory import build_llm, build_embeddings


class RagInput(BaseModel):
    question: str = Field(..., description="User question about Promtior")


def _load_retriever():
    embeddings = build_embeddings()
    vectordb = Chroma(
        collection_name="promtior_site",
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )
    return vectordb.as_retriever(search_kwargs={"k": 4})


def _format_docs(docs) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        parts.append(f"[source: {src}] {d.page_content}")
    return "\n\n".join(parts)


def build_rag_chain():
    retriever = _load_retriever()
    llm = build_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful assistant. Answer ONLY using the provided context. "
             "If the context is insufficient, say you don't know and suggest what to ingest or where to look."),
            ("human",
             "Question:\n{question}\n\n"
             "Context:\n{context}\n\n"
             "Answer in a clear, concise way and cite sources by URL when relevant."),
        ]
    )

    # ✅ Always turn the input into a plain string question
    def extract_question(x):
        # When called by LangServe, x is usually a dict: {"question": "..."}
        if isinstance(x, dict) and "question" in x:
            return x["question"]
        # If it is a Pydantic model (rare depending on versions)
        if hasattr(x, "question"):
            return x.question
        # Last fallback (avoid crashing)
        return str(x)

    get_question = RunnableLambda(extract_question)

    chain = (
        {
            "question": get_question,
            "context": get_question | retriever | RunnableLambda(_format_docs),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain




rag_chain = build_rag_chain()
