# Component Diagram (Mermaid)

```mermaid
flowchart LR
  U[User] -->|Question| API[LangServe / FastAPI]
  API -->|invoke| RAG[RAG Chain]
  RAG -->|retrieve top-k| VDB[(Chroma Vector DB)]
  VDB -->|chunks + sources| RAG
  RAG -->|prompt with context| LLM[LLM
(OpenAI or Ollama)]
  LLM -->|answer| API
  API -->|response| U

  subgraph Offline/Build Step
    C[Crawler
(promtior.ai)] --> T[HTML->Text Cleaner]
    T --> S[Text Splitter]
    S --> E[Embeddings]
    E --> VDB
  end
```
