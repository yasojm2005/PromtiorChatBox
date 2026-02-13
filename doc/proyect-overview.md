Project Overview
Objective

The objective of this project was to design and implement an intelligent chatbot assistant capable of answering questions related to the Promtior website content. The solution was required to leverage Retrieval-Augmented Generation (RAG) using LangChain and be exposed through LangServe, ensuring scalability, modularity, and production readiness.

Approach and Solution Design

To solve the challenge, I designed a RAG-based architecture that combines document retrieval with large language model generation.

The implementation was divided into four major stages:

1️⃣ Data Ingestion

The chatbot knowledge base is built by ingesting data from multiple sources:

Promtior public website pages (HTML crawling)

PDF documentation (technical test material and company documents)

A custom crawler downloads website content, removes HTML noise (navigation bars, scripts, footers), and converts pages into clean text.

PDF files are parsed using a document loader that extracts text page by page.

2️⃣ Text Processing and Chunking

Since LLMs and vector databases work better with smaller semantic units, the content is split into overlapping text chunks using a recursive character splitter.

Configuration used:

Chunk size: 1000 characters

Overlap: 150 characters

This ensures semantic continuity between chunks while optimizing retrieval accuracy.

3️⃣ Embedding and Vector Storage

Each chunk is transformed into vector embeddings using OpenAI’s embedding model.

These vectors are stored in a persistent Chroma vector database, enabling semantic similarity search.

Storage configuration:

Local development: ./data/chroma

Cloud deployment: persistent Railway volume

4️⃣ Retrieval-Augmented Generation (RAG)

When a user asks a question:

The query is embedded.

The retriever performs similarity search in Chroma.

Top-K relevant chunks are retrieved.

The context is injected into a prompt.

The LLM generates a grounded response.

This ensures answers are based on company data rather than hallucinated knowledge.

API Exposure with LangServe

The RAG chain is exposed as REST endpoints using LangServe on top of FastAPI.

Available endpoints include:

/rag/invoke → synchronous responses

/rag/stream → token streaming

/rag/playground → interactive UI

This enables integration with external applications or front-end chat interfaces.

Deployment Strategy

The solution is containerized using Docker and deployed on Railway cloud infrastructure.

Key deployment considerations:

Dynamic port binding via environment variable PORT

Environment variable configuration for API keys

Persistent storage for vector database

Scalable HTTP API exposure

Challenges Encountered and Solutions
Challenge	Solution
Website crawling limited to few pages	Adjusted crawler depth and parsing logic
Embedding quota limits	Configured OpenAI billing and fallback to Ollama
Dependency conflicts (httpx, langserve)	Pinned compatible library versions
Dict vs string retrieval error	Implemented input extraction layer in chain
Port conflicts in deployment	Added dynamic port binding
Result

The final solution is a production-ready chatbot capable of:

Answering questions about Promtior

Retrieving knowledge from web + PDF sources

Providing grounded responses with context

Scaling via cloud deployment