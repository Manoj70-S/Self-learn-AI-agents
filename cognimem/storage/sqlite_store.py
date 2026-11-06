import sqlite3
import json
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime, timezone
import numpy as np

from cognimem.models import MemoryItem, MemoryType, MemoryStatus
from cognimem.storage.base import BaseVectorStore


class SQLiteVectorStore(BaseVectorStore):
    """
    Self-contained, zero-dependency persistent Vector Store using SQLite and NumPy.
    Stores raw float32 vector blobs and performs high-speed vectorized cosine similarity.
    """

    def __init__(self, db_path: str = "cognimem.db"):
        self.db_path = db_path
        self._mem_conn = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        self.initialize()

    def _get_connection(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize(self) -> None:
        """Create schema and indexes if not existing."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    agent_id TEXT,
                    session_id TEXT,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    superseded_by TEXT,
                    metadata_json TEXT,
                    embedding_blob BLOB
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_status ON memories(user_id, status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at);")
            conn.commit()

    def _row_to_item(self, row: sqlite3.Row, include_embedding: bool = False) -> MemoryItem:
        embedding = None
        if include_embedding and row["embedding_blob"]:
            embedding = np.frombuffer(row["embedding_blob"], dtype=np.float32).tolist()

        return MemoryItem(
            id=row["id"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            session_id=row["session_id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            status=MemoryStatus(row["status"]),
            confidence=float(row["confidence"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None,
            access_count=int(row["access_count"]),
            superseded_by=row["superseded_by"],
            metadata=json.loads(row["metadata_json"]) if row["metadata_json"] else {},
            embedding=embedding
        )

    def upsert(self, item: MemoryItem) -> None:
        emb_blob = None
        if item.embedding:
            emb_blob = np.array(item.embedding, dtype=np.float32).tobytes()

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO memories (
                    id, user_id, agent_id, session_id, content, memory_type, status,
                    confidence, created_at, updated_at, last_accessed_at, access_count,
                    superseded_by, metadata_json, embedding_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content=excluded.content,
                    memory_type=excluded.memory_type,
                    status=excluded.status,
                    confidence=excluded.confidence,
                    updated_at=excluded.updated_at,
                    last_accessed_at=excluded.last_accessed_at,
                    access_count=excluded.access_count,
                    superseded_by=excluded.superseded_by,
                    metadata_json=excluded.metadata_json,
                    embedding_blob=coalesce(excluded.embedding_blob, memories.embedding_blob);
            """, (
                item.id,
                item.user_id,
                item.agent_id,
                item.session_id,
                item.content,
                item.memory_type.value if hasattr(item.memory_type, 'value') else str(item.memory_type),
                item.status.value if hasattr(item.status, 'value') else str(item.status),
                item.confidence,
                item.created_at.isoformat(),
                item.updated_at.isoformat(),
                item.last_accessed_at.isoformat() if item.last_accessed_at else None,
                item.access_count,
                item.superseded_by,
                json.dumps(item.metadata),
                emb_blob
            ))
            conn.commit()

    def upsert_batch(self, items: List[MemoryItem]) -> None:
        for item in items:
            self.upsert(item)

    def get(self, memory_id: str) -> Optional[MemoryItem]:
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM memories WHERE id = ?;", (memory_id,))
            row = cur.fetchone()
            return self._row_to_item(row, include_embedding=True) if row else None

    def get_all(self, user_id: str, status: Optional[MemoryStatus] = None) -> List[MemoryItem]:
        with self._get_connection() as conn:
            if status is not None:
                val = status.value if hasattr(status, 'value') else str(status)
                cur = conn.execute(
                    "SELECT * FROM memories WHERE user_id = ? AND status = ? ORDER BY created_at DESC;",
                    (user_id, val)
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC;",
                    (user_id,)
                )
            return [self._row_to_item(r) for r in cur.fetchall()]

    def search(
        self,
        query_vector: List[float],
        user_id: str,
        limit: int = 5,
        threshold: float = 0.0,
        status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> List[Tuple[MemoryItem, float]]:
        status_val = status.value if hasattr(status, 'value') else str(status)
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT * FROM memories WHERE user_id = ? AND status = ? AND embedding_blob IS NOT NULL;",
                (user_id, status_val)
            )
            rows = cur.fetchall()

        if not rows:
            return []

        q_vec = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        # Vectorized similarity computation
        scored: List[Tuple[MemoryItem, float]] = []
        for r in rows:
            mem_vec = np.frombuffer(r["embedding_blob"], dtype=np.float32)
            m_norm = np.linalg.norm(mem_vec)
            if m_norm > 0:
                similarity = float(np.dot(q_vec, mem_vec) / (q_norm * m_norm))
            else:
                similarity = 0.0

            if similarity >= threshold:
                item = self._row_to_item(r)
                scored.append((item, similarity))

        # Sort descending by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def update_status(
        self,
        memory_id: str,
        status: MemoryStatus,
        superseded_by: Optional[str] = None
    ) -> bool:
        now_str = datetime.now(timezone.utc).isoformat()
        status_val = status.value if hasattr(status, 'value') else str(status)
        with self._get_connection() as conn:
            cur = conn.execute("""
                UPDATE memories
                SET status = ?, superseded_by = ?, updated_at = ?
                WHERE id = ?;
            """, (status_val, superseded_by, now_str, memory_id))
            conn.commit()
            return cur.rowcount > 0

    def delete(self, memory_id: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id = ?;", (memory_id,))
            conn.commit()
            return cur.rowcount > 0

    def clear(self, user_id: Optional[str] = None) -> int:
        with self._get_connection() as conn:
            if user_id:
                cur = conn.execute("DELETE FROM memories WHERE user_id = ?;", (user_id,))
            else:
                cur = conn.execute("DELETE FROM memories;")
            conn.commit()
            return cur.rowcount
