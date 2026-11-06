from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from cognimem.models import MemoryItem, MemoryStatus


class BaseVectorStore(ABC):
    """Abstract interface for memory persistence and vector indexing."""

    @abstractmethod
    def initialize(self) -> None:
        """Create necessary collections, tables, or indexes."""
        pass

    @abstractmethod
    def upsert(self, item: MemoryItem) -> None:
        """Insert or update a memory item."""
        pass

    @abstractmethod
    def upsert_batch(self, items: List[MemoryItem]) -> None:
        """Bulk insert or update memory items."""
        pass

    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryItem]:
        """Fetch memory item by ID."""
        pass

    @abstractmethod
    def get_all(self, user_id: str, status: Optional[MemoryStatus] = None) -> List[MemoryItem]:
        """List all memories belonging to a specific user, optionally filtered by status."""
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        user_id: str,
        limit: int = 5,
        threshold: float = 0.0,
        status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Search memories by semantic similarity.
        Returns list of (MemoryItem, similarity_score) sorted by score descending.
        """
        pass

    @abstractmethod
    def update_status(
        self,
        memory_id: str,
        status: MemoryStatus,
        superseded_by: Optional[str] = None
    ) -> bool:
        """Update the lifecycle status of a memory."""
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> bool:
        """Permanently remove a memory item."""
        pass

    @abstractmethod
    def clear(self, user_id: Optional[str] = None) -> int:
        """Purge memories for a user or entire store."""
        pass
