"""
FAISS-based RAG for real estate market knowledge retrieval.
Uses sentence-transformers for embeddings (no OpenAI key needed).
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config import EMBEDDING_MODEL, FAISS_INDEX_PATH, TOP_K_RETRIEVAL
from knowledge_base import MARKET_DOCUMENTS


class RealEstateRAG:
    """FAISS-backed RAG system for real estate knowledge retrieval."""

    def __init__(self):
        print("🔍 Initializing RAG system...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = self._load_or_build_index()
        self.retriever   = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_RETRIEVAL}
        )
        print("✅ RAG system ready")

    def _load_or_build_index(self) -> FAISS:
        if os.path.exists(FAISS_INDEX_PATH):
            print("   Loading cached FAISS index...")
            return FAISS.load_local(
                FAISS_INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        return self._build_index()

    def _build_index(self) -> FAISS:
        print("   Building FAISS index from knowledge base...")
        docs = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=64
        )

        for doc in MARKET_DOCUMENTS:
            chunks = splitter.split_text(doc["text"].strip())
            for chunk in chunks:
                docs.append(Document(
                    page_content=chunk,
                    metadata={
                        "id":       doc["id"],
                        "city":     doc["city"],
                        "category": doc["category"]
                    }
                ))

        vectorstore = FAISS.from_documents(docs, self.embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"   ✅ Index built with {len(docs)} chunks")
        return vectorstore

    def retrieve(self, query: str, city: str = None, k: int = None) -> str:
        """Retrieve and format relevant market context."""
        k = k or TOP_K_RETRIEVAL

        if city:
            enriched_query = f"{city} real estate: {query}"
        else:
            enriched_query = query

        results = self.vectorstore.similarity_search(enriched_query, k=k)

        seen_cats = set()
        unique_results = []
        for r in results:
            cat = r.metadata.get("category", "")
            if cat not in seen_cats:
                unique_results.append(r)
                seen_cats.add(cat)

        if not unique_results:
            return "No specific market context found."

        formatted = []
        for i, r in enumerate(unique_results, 1):
            source = f"[{r.metadata.get('city','All')} | {r.metadata.get('category','')}]"
            formatted.append(f"Source {i} {source}:\n{r.page_content}")

        return "\n\n".join(formatted)

    def retrieve_city_specific(self, city: str) -> str:
        """Retrieve all documents specific to a city."""
        results = self.vectorstore.similarity_search(
            f"{city} rental market investment trends",
            k=6
        )
        city_results = [
            r for r in results
            if r.metadata.get("city") in (city, "All")
        ]
        if not city_results:
            city_results = results[:3]

        return "\n\n".join(r.page_content for r in city_results)


_rag_instance = None

def get_rag() -> RealEstateRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RealEstateRAG()
    return _rag_instance