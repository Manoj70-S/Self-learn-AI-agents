from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
import json
import uuid

from cognimem.models import MemoryItem, MemoryType, MemoryStatus
from cognimem.storage.base import BaseVectorStore


class QdrantVectorStore(BaseVectorStore):
    """
    Enterprise-scale Vector Store using Qdrant.
    Leverages HNSW indexing and payload filtering on user_id and status.
    """

    def __init__(
        self,
        collection_name: str = "agent_memories",
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        vector_size: int = 1536
    ):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models as qmodels
        except ImportError:
            raise ImportError(
                "qdrant-client is not installed. Run `pip install qdrant-client` to use QdrantVectorStore."
            )

        self.collection_name = collection_name
        self.vector_size = vector_size
        self.qmodels = qmodels

        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port, api_key=api_key)

        self.initialize()

    def initialize(self) -> None:
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=self.qmodels.VectorParams(
                    size=self.vector_size,
                    distance=self.qmodels.Distance.COSINE
                )
            )

        # Ensure payload indices exist for filtering
        for field in ("user_id", "status"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=self.qmodels.PayloadSchemaType.KEYWORD
                )
            except Exception:
                pass

    def _payload_to_item(self, point_id: str, payload: Dict[str, Any], vector: Optional[List[float]] = None) -> MemoryItem:
        return MemoryItem(
            id=point_id,
            user_id=payload["user_id"],
            agent_id=payload.get("agent_id"),
            session_id=payload.get("session_id"),
            content=payload["content"],
            memory_type=MemoryType(payload["memory_type"]),
            status=MemoryStatus(payload["status"]),
            confidence=float(payload.get("confidence", 1.0)),
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            last_accessed_at=datetime.fromisoformat(payload["last_accessed_at"]) if payload.get("last_accessed_at") else None,
            access_count=int(payload.get("access_count", 0)),
            superseded_by=payload.get("superseded_by"),
            metadata=payload.get("metadata", {}),
            embedding=vector
        )

    def upsert(self, item: MemoryItem) -> None:
        point_id = item.id
        try:
            uuid.UUID(point_id)
        except ValueError:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item.id))

        payload = {
            "user_id": item.user_id,
            "agent_id": item.agent_id,
            "session_id": item.session_id,
            "content": item.content,
            "memory_type": item.memory_type.value if hasattr(item.memory_type, "value") else str(item.memory_type),
            "status": item.status.value if hasattr(item.status, "value") else str(item.status),
            "confidence": item.confidence,
            "created_at": item.created_at.isoformat(),
            "updated_at": item.updated_at.isoformat(),
            "last_accessed_at": item.last_accessed_at.isoformat() if item.last_accessed_at else None,
            "access_count": item.access_count,
            "superseded_by": item.superseded_by,
            "metadata": item.metadata,
        }

        # If no embedding is provided (e.g. telemetry access update), update payload only
        if not item.embedding:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=[point_id]
            )
            return

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                self.qmodels.PointStruct(
                    id=point_id,
                    vector=item.embedding,
                    payload=payload
                )
            ]
        )

    def upsert_batch(self, items: List[MemoryItem]) -> None:
        points = []
        for item in items:
            if not item.embedding:
                continue
            point_id = item.id
            try:
                uuid.UUID(point_id)
            except ValueError:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item.id))

            payload = {
                "user_id": item.user_id,
                "agent_id": item.agent_id,
                "session_id": item.session_id,
                "content": item.content,
                "memory_type": item.memory_type.value if hasattr(item.memory_type, "value") else str(item.memory_type),
                "status": item.status.value if hasattr(item.status, "value") else str(item.status),
                "confidence": item.confidence,
                "created_at": item.created_at.isoformat(),
                "updated_at": item.updated_at.isoformat(),
                "last_accessed_at": item.last_accessed_at.isoformat() if item.last_accessed_at else None,
                "access_count": item.access_count,
                "superseded_by": item.superseded_by,
                "metadata": item.metadata,
            }
            points.append(self.qmodels.PointStruct(id=point_id, vector=item.embedding, payload=payload))

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        point_id = memory_id
        try:
            uuid.UUID(point_id)
        except ValueError:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, memory_id))

        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id],
            with_vectors=True,
            with_payload=True
        )
        if not results:
            return None
        pt = results[0]
        return self._payload_to_item(str(pt.id), pt.payload, pt.vector)

    def get_all(self, user_id: str, status: Optional[MemoryStatus] = None) -> List[MemoryItem]:
        conditions = [
            self.qmodels.FieldCondition(
                key="user_id",
                match=self.qmodels.MatchValue(value=user_id)
            )
        ]
        if status is not None:
            val = status.value if hasattr(status, "value") else str(status)
            conditions.append(
                self.qmodels.FieldCondition(
                    key="status",
                    match=self.qmodels.MatchValue(value=val)
                )
            )

        scroll_filter = self.qmodels.Filter(must=conditions)
        records, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=1000,
            with_payload=True,
            with_vectors=False
        )

        return [self._payload_to_item(str(r.id), r.payload) for r in records]

    def search(
        self,
        query_vector: List[float],
        user_id: str,
        limit: int = 5,
        threshold: float = 0.0,
        status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> List[Tuple[MemoryItem, float]]:
        status_val = status.value if hasattr(status, "value") else str(status)
        search_filter = self.qmodels.Filter(
            must=[
                self.qmodels.FieldCondition(key="user_id", match=self.qmodels.MatchValue(value=user_id)),
                self.qmodels.FieldCondition(key="status", match=self.qmodels.MatchValue(value=status_val))
            ]
        )

        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=threshold if threshold > 0 else None,
                with_payload=True
            )
            search_result = response.points
        else:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=threshold if threshold > 0 else None,
                with_payload=True
            )

        results: List[Tuple[MemoryItem, float]] = []
        for hit in search_result:
            item = self._payload_to_item(str(hit.id), hit.payload)
            results.append((item, float(hit.score)))
        return results

    def update_status(
        self,
        memory_id: str,
        status: MemoryStatus,
        superseded_by: Optional[str] = None
    ) -> bool:
        point_id = memory_id
        try:
            uuid.UUID(point_id)
        except ValueError:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, memory_id))

        status_val = status.value if hasattr(status, "value") else str(status)
        now_str = datetime.now(timezone.utc).isoformat()

        self.client.set_payload(
            collection_name=self.collection_name,
            payload={
                "status": status_val,
                "superseded_by": superseded_by,
                "updated_at": now_str
            },
            points=[point_id]
        )
        return True

    def delete(self, memory_id: str) -> bool:
        point_id = memory_id
        try:
            uuid.UUID(point_id)
        except ValueError:
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, memory_id))

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self.qmodels.PointIdsList(points=[point_id])
        )
        return True

    def clear(self, user_id: Optional[str] = None) -> int:
        if user_id:
            filt = self.qmodels.Filter(
                must=[self.qmodels.FieldCondition(key="user_id", match=self.qmodels.MatchValue(value=user_id))]
            )
            self.client.delete(collection_name=self.collection_name, points_selector=self.qmodels.FilterSelector(filter=filt))
            return 1
        else:
            self.client.delete_collection(collection_name=self.collection_name)
            self.initialize()
            return 1
