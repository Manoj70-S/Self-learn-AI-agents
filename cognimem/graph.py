from typing import List, Dict, Any, Optional, Set
import sqlite3
from datetime import datetime, timezone
import networkx as nx

from cognimem.models import EntityRelation


class KnowledgeGraph:
    """
    Relational Knowledge Graph memory layer.
    Extracts and links (Subject, Predicate, Object) triples to enable multi-hop reasoning.
    Backed by persistent SQLite and in-memory NetworkX graph analysis.
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
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT NOT NULL
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_user ON graph_triples(user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_subj ON graph_triples(subject);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_obj ON graph_triples(object);")
            conn.commit()

    def add_relation(
        self,
        subject: str,
        predicate: str,
        obj: str,
        user_id: str,
        confidence: float = 1.0
    ) -> None:
        s_norm = subject.strip()
        p_norm = predicate.strip().upper()
        o_norm = obj.strip()
        now_str = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            # Check for existing identical triple
            cur = conn.execute(
                "SELECT id FROM graph_triples WHERE user_id = ? AND subject = ? AND predicate = ? AND object = ?;",
                (user_id, s_norm, p_norm, o_norm)
            )
            if cur.fetchone() is None:
                conn.execute("""
                    INSERT INTO graph_triples (user_id, subject, predicate, object, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?);
                """, (user_id, s_norm, p_norm, o_norm, confidence, now_str))
                conn.commit()

    def get_relations(self, user_id: str) -> List[EntityRelation]:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT subject, predicate, object, confidence, created_at FROM graph_triples WHERE user_id = ? ORDER BY id DESC;",
                (user_id,)
            )
            rows = cur.fetchall()

        return [
            EntityRelation(
                subject=r["subject"],
                predicate=r["predicate"],
                object=r["object"],
                confidence=float(r["confidence"]),
                created_at=datetime.fromisoformat(r["created_at"])
            )
            for r in rows
        ]

    def build_networkx_graph(self, user_id: str) -> nx.MultiDiGraph:
        """Constructs an in-memory directed multi-graph for graph traversal."""
        G = nx.MultiDiGraph()
        relations = self.get_relations(user_id)
        for rel in relations:
            G.add_edge(rel.subject, rel.object, relation=rel.predicate, confidence=rel.confidence)
        return G

    def get_connected_entities(self, entity: str, user_id: str, max_hops: int = 2) -> List[Dict[str, Any]]:
        """Multi-hop search: finds all entities related to the given entity."""
        G = self.build_networkx_graph(user_id)
        if entity not in G:
            # Case-insensitive match attempt
            matched = [n for n in G.nodes() if n.lower() == entity.lower()]
            if not matched:
                return []
            entity = matched[0]

        connected: List[Dict[str, Any]] = []
        visited: Set[str] = {entity}
        queue = [(entity, 0)]

        while queue:
            curr, depth = queue.pop(0)
            if depth >= max_hops:
                continue

            # Forward edges
            for nbr in G.successors(curr):
                for edge_key, data in G[curr][nbr].items():
                    connected.append({
                        "source": curr,
                        "relation": data.get("relation", "CONNECTED_TO"),
                        "target": nbr,
                        "hops": depth + 1
                    })
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append((nbr, depth + 1))

            # Backward edges
            for predecessor in G.predecessors(curr):
                for edge_key, data in G[predecessor][curr].items():
                    connected.append({
                        "source": predecessor,
                        "relation": data.get("relation", "CONNECTED_TO"),
                        "target": curr,
                        "hops": depth + 1
                    })
                if predecessor not in visited:
                    visited.add(predecessor)
                    queue.append((predecessor, depth + 1))

        return connected

    def export_elements_for_ui(self, user_id: str) -> Dict[str, List[Dict[str, Any]]]:
        """Exports graph format ready for visual rendering (e.g. Streamlit or Cytoscape)."""
        relations = self.get_relations(user_id)
        nodes_set = set()
        edges = []

        for r in relations:
            nodes_set.add(r.subject)
            nodes_set.add(r.object)
            edges.append({
                "source": r.subject,
                "target": r.object,
                "label": r.predicate
            })

        nodes = [{"id": n, "label": n} for n in nodes_set]
        return {"nodes": nodes, "edges": edges}

    def clear(self, user_id: Optional[str] = None) -> int:
        with self._get_connection() as conn:
            if user_id:
                cur = conn.execute("DELETE FROM graph_triples WHERE user_id = ?;", (user_id,))
            else:
                cur = conn.execute("DELETE FROM graph_triples;")
            conn.commit()
            return cur.rowcount
