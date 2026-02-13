"""
FastAPI + LangServe server.

Run:
  python -m app.server
"""
from __future__ import annotations

from fastapi import FastAPI
from langserve import add_routes

from .chain import rag_chain, RagInput

app = FastAPI(title="Promtior RAG Chatbot")

# Exposes:
#  - POST /rag/invoke
#  - POST /rag/stream
#  - OpenAPI docs at /docs
add_routes(app, rag_chain, path="/rag", input_type=RagInput)

if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

