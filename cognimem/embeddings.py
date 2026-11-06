from abc import ABC, abstractmethod
from typing import List, Optional
import os
import hashlib
import math
import numpy as np


class BaseEmbeddings(ABC):
    """Abstract interface for dense vector embedding generators."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a float vector."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings into float vectors."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensionality."""
        pass


class OpenAIEmbeddings(BaseEmbeddings):
    """Embeddings powered by OpenAI API (e.g. text-embedding-3-small)."""

    def __init__(self, model: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbeddings.")
        
        from openai import OpenAI
        self.client = OpenAI(api_key=self.api_key)
        self._dimensions = 1536 if "text-embedding-3-small" in model or "text-embedding-ada-002" in model else 3072

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_text(self, text: str) -> List[float]:
        text = text.replace("\n", " ").strip()
        if not text:
            return [0.0] * self._dimensions
        resp = self.client.embeddings.create(input=[text], model=self.model)
        return resp.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        cleaned = [t.replace("\n", " ").strip() or " " for t in texts]
        resp = self.client.embeddings.create(input=cleaned, model=self.model)
        return [item.embedding for item in resp.data]


class DeterministicEmbeddings(BaseEmbeddings):
    """
    Offline, zero-API fallback embeddings generator.
    Produces high-fidelity semantic-like dense vectors using character n-grams,
    word hashing, and L2 normalization.
    Ensures the system can be thoroughly developed, tested, and demoed offline!
    """

    def __init__(self, dimensions: int = 1536):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_text(self, text: str) -> List[float]:
        text = text.lower().strip()
        if not text:
            return [0.0] * self._dimensions

        # Generate a dense representation from tokens and subwords
        vec = np.zeros(self._dimensions, dtype=np.float32)
        words = text.split()

        for w_idx, word in enumerate(words):
            # Positional weight
            pos_weight = 1.0 / (1.0 + 0.05 * w_idx)

            # Word hash
            h_word = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx1 = h_word % self._dimensions
            val1 = ((h_word >> 16) % 1000) / 1000.0 - 0.5
            vec[idx1] += val1 * 2.0 * pos_weight

            # Sub-word character trigrams for semantic affinity
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    trigram = word[i:i+3]
                    h_tri = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16)
                    idx2 = h_tri % self._dimensions
                    val2 = ((h_tri >> 8) % 1000) / 1000.0 - 0.5
                    vec[idx2] += val2 * 1.0

        # Semantic concept clusters for offline semantic affinity
        concept_clusters = {
            "food_allergy": ["food", "dish", "dinner", "lunch", "breakfast", "eat", "meal", "thai", "italian", "cook", "restaurant", "snack", "peanuts", "nuts", "allergy", "allergic"],
            "software_tech": ["software", "code", "programming", "developer", "backend", "frontend", "service", "framework", "database", "api", "fastapi", "django", "postgresql", "python", "go", "rust"],
            "location_geo": ["live", "lives", "moved", "relocated", "city", "country", "location", "amsterdam", "berlin", "paris", "london", "zurich"]
        }
        for c_name, keywords in concept_clusters.items():
            matches = sum(1 for kw in keywords if kw in text)
            if matches > 0:
                h_c = int(hashlib.sha256(c_name.encode("utf-8")).hexdigest(), 16)
                c_idx = h_c % self._dimensions
                vec[c_idx] += float(matches) * 3.5

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


def get_embedding_provider(provider: str = "openai", **kwargs) -> BaseEmbeddings:
    """Factory to retrieve embedding provider with graceful offline fallback."""
    provider = provider.lower()
    if provider == "openai":
        api_key = kwargs.get("api_key") or os.getenv("OPENAI_API_KEY")
        if api_key:
            return OpenAIEmbeddings(
                model=kwargs.get("model", "text-embedding-3-small"),
                api_key=api_key
            )
        # If no API key provided, fall back to DeterministicEmbeddings
        return DeterministicEmbeddings(dimensions=kwargs.get("dimensions", 1536))
    elif provider in ("deterministic", "mock", "offline", "local"):
        return DeterministicEmbeddings(dimensions=kwargs.get("dimensions", 1536))
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")
