# Project Overview

## Goal
Build a chatbot assistant using **RAG (Retrieval Augmented Generation)** to answer questions about the content of the Promtior website, implemented with **LangChain** and exposed via **LangServe**.

## Approach (high level)
1. **Ingestion**: crawl Promtior pages, convert HTML → clean text, save a raw cache.
2. **Chunking**: split long text into overlapping chunks (1,000 chars, 150 overlap).
3. **Embeddings + Vector DB**: embed each chunk and store it in **Chroma** (persistent local directory).
4. **Retrieval**: for each user question, retrieve top-k relevant chunks.
5. **Generation**: call the LLM with a grounded prompt that instructs it to answer only using retrieved context and cite URLs.

## Challenges and solutions
- **Website noise**: headers/footers/scripts can pollute retrieval → remove common noisy tags and collapse whitespace.
- **Freshness**: website can change → re-run ingestion to rebuild the index.
- **Model choice**: the project supports both **OpenAI** and **Ollama** via environment variables, so it can run in local/offline scenarios.

## How to demonstrate requirement questions
After ingestion and running the server, query:
- "What services does Promtior offer?"
- "When was the company founded?"

## Improvements (if you want extra points)
- Add more sources (e.g., this PDF/presentation) as an additional loader.
- Use sitemap-based crawling if available.
- Add reranking (e.g., Cohere or bge-reranker) for better retrieval quality.
