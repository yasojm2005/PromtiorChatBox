"""
Ingest Promtior website pages -> chunk -> embed -> store in Chroma.

Run:
  python -m app.ingest
"""
from __future__ import annotations

import os
import re
import time
import json
import hashlib
from dataclasses import dataclass
from urllib.parse import urljoin, urldefrag, urlparse

import requests
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from .config import settings
from .llm_factory import build_embeddings


def _normalize_url(url: str) -> str:
    url, _frag = urldefrag(url)
    return url.strip()


def _is_same_domain(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _page_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # Remove nav/script/style/footer to reduce noise
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Common noisy sections
    for selector in ["header", "nav", "footer"]:
        for tag in soup.select(selector):
            tag.decompose()

    text = soup.get_text(" ", strip=True)
    return _clean_text(text)


@dataclass
class CrawlResult:
    url: str
    text: str


def crawl_site(base_url: str, max_pages: int, max_depth: int, timeout: int) -> list[CrawlResult]:
    seen: set[str] = set()
    queue: list[tuple[str, int]] = [(base_url, 0)]
    out: list[CrawlResult] = []

    session = requests.Session()
    session.headers.update({"User-Agent": "promtior-rag-bot/1.0 (+https://example.com)"})

    while queue and len(out) < max_pages:
        url, depth = queue.pop(0)
        url = _normalize_url(url)

        if url in seen:
            continue
        seen.add(url)

        if depth > max_depth:
            continue
        if not _is_same_domain(url, base_url):
            continue

        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            html = resp.text
            text = _page_to_text(html)

            if len(text) < 200:
                # skip near-empty pages
                continue

            out.append(CrawlResult(url=url, text=text))

            # Extract links for crawling
            soup = BeautifulSoup(html, "lxml")
            for a in soup.find_all("a", href=True):
                href = a.get("href")
                if not href:
                    continue
                next_url = _normalize_url(urljoin(url, href))
                if _is_same_domain(next_url, base_url) and next_url not in seen:
                    queue.append((next_url, depth + 1))

            time.sleep(0.2)  # be polite
        except Exception as e:
            print(f"[crawl] Skipping {url}: {e}")

    return out


def write_raw_cache(results: list[CrawlResult], raw_dir: str) -> None:
    os.makedirs(raw_dir, exist_ok=True)
    index = []
    for r in results:
        h = hashlib.sha256(r.url.encode("utf-8")).hexdigest()[:16]
        path = os.path.join(raw_dir, f"{h}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)
        index.append({"url": r.url, "file": os.path.basename(path), "chars": len(r.text)})
    with open(os.path.join(raw_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def build_documents(results: list[CrawlResult]) -> list[Document]:
    docs: list[Document] = []
    for r in results:
        docs.append(Document(page_content=r.text, metadata={"source": r.url}))
    return docs


def main() -> None:
    print("== Promtior RAG ingestion ==")
    print(f"Base URL: {settings.promtior_base_url}")
    print(f"Max pages: {settings.crawl_max_pages}, max depth: {settings.crawl_max_depth}")
    print(f"Chroma dir: {settings.chroma_dir}")
    print(f"Raw dir: {settings.raw_dir}")

    results = crawl_site(
        base_url=settings.promtior_base_url,
        max_pages=settings.crawl_max_pages,
        max_depth=settings.crawl_max_depth,
        timeout=settings.request_timeout_secs,
    )
    print(f"[crawl] Collected {len(results)} pages")

    write_raw_cache(results, settings.raw_dir)

    docs = build_documents(results)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"[chunk] Produced {len(chunks)} chunks")

    embeddings = build_embeddings()

    os.makedirs(settings.chroma_dir, exist_ok=True)
    vectordb = Chroma(
        collection_name="promtior_site",
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )

    # reset collection
    try:
        vectordb.delete_collection()
    except Exception:
        pass

    vectordb = Chroma(
        collection_name="promtior_site",
        embedding_function=embeddings,
        persist_directory=settings.chroma_dir,
    )

    vectordb.add_documents(chunks)
    vectordb.persist()
    print("[done] Vector store persisted.")


if __name__ == "__main__":
    main()
