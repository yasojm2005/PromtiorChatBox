import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    provider: str = os.getenv("PROVIDER", "openai").strip().lower()

    # OpenAI
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None

    # Ollama
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_llm_model: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.1")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    # RAG / storage
    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")
    raw_dir: str = os.getenv("RAW_DIR", "./data/raw")

    # Crawl settings
    promtior_base_url: str = os.getenv("PROMTIOR_BASE_URL", "https://promtior.ai/")
    crawl_max_pages: int = int(os.getenv("CRAWL_MAX_PAGES", "25"))
    crawl_max_depth: int = int(os.getenv("CRAWL_MAX_DEPTH", "2"))
    request_timeout_secs: int = int(os.getenv("REQUEST_TIMEOUT_SECS", "15"))

settings = Settings()
