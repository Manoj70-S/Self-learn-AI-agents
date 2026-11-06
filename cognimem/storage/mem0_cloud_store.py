from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
import os

from cognimem.models import MemoryItem, MemoryType, MemoryStatus
from cognimem.storage.base import BaseVectorStore


class Mem0CloudStore(BaseVectorStore):
    """
    Cloud-managed Vector Store backed by the official Mem0 Platform API.
    Uses the provided Mem0 API key for automated indexing and extraction.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MEM0_API_KEY")
        if not self.api_key:
            raise ValueError("MEM0_API_KEY is required to initialize Mem0CloudStore.")

        try:
            from mem0 import MemoryClient
            self.client = MemoryClient(api_key=self.api_key)
        except ImportError:
            raise ImportError("mem0ai package is required. Run `pip install mem0ai`.")

    def initialize(self) -> None:
        pass

    def upsert(self, item: MemoryItem) -> None:
        self.client.add(item.content, user_id=item.user_id)

    def upsert_batch(self, items: List[MemoryItem]) -> None:
        for item in items:
            self.upsert(item)

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        try:
            res = self.client.get(memory_id)
            if not res:
                return None
            return MemoryItem(
                id=res.get("id", memory_id),
                user_id=res.get("user_id", "default"),
                content=res.get("memory", ""),
                memory_type=MemoryType.PREFERENCE,
                status=MemoryStatus.ACTIVE
            )
        except Exception:
            return None

    def get_all(self, user_id: str, status: Optional[MemoryStatus] = None) -> List[MemoryItem]:
        res = self.client.get_all(filters={"user_id": user_id})
        items: List[MemoryItem] = []
        for mem in res.get("results", []):
            items.append(MemoryItem(
                id=mem.get("id"),
                user_id=user_id,
                content=mem.get("memory", ""),
                memory_type=MemoryType.PREFERENCE,
                status=MemoryStatus.ACTIVE,
                created_at=datetime.fromisoformat(mem.get("created_at")) if mem.get("created_at") else datetime.now(timezone.utc)
            ))
        return items

    def search(
        self,
        query_vector: List[float],
        user_id: str,
        limit: int = 5,
        threshold: float = 0.0,
        status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> List[Tuple[MemoryItem, float]]:
        # Mem0 Cloud search uses query text rather than raw vectors
        res = self.client.get_all(filters={"user_id": user_id})
        items = []
        for mem in res.get("results", [])[:limit]:
            item = MemoryItem(
                id=mem.get("id"),
                user_id=user_id,
                content=mem.get("memory", ""),
                memory_type=MemoryType.PREFERENCE,
                status=MemoryStatus.ACTIVE
            )
            items.append((item, 1.0))
        return items

    def update_status(
        self,
        memory_id: str,
        status: MemoryStatus,
        superseded_by: Optional[str] = None
    ) -> bool:
        if status in (MemoryStatus.ARCHIVED, MemoryStatus.SUPERSEDED):
            return self.delete(memory_id)
        return True

    def delete(self, memory_id: str) -> bool:
        try:
            self.client.delete(memory_id)
            return True
        except Exception:
            return False

    def clear(self, user_id: Optional[str] = None) -> int:
        try:
            if user_id:
                self.client.delete_all(filters={"user_id": user_id})
            else:
                self.client.delete_all()
            return 1
        except Exception:
            return 0
