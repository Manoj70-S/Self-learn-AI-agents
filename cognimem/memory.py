from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
import math
import uuid

from cognimem.models import (
    MemoryItem,
    MemoryType,
    MemoryStatus,
    ReconciliationAction,
)
from cognimem.storage.base import BaseVectorStore
from cognimem.storage.sqlite_store import SQLiteVectorStore
from cognimem.embeddings import BaseEmbeddings, get_embedding_provider
from cognimem.extractor import CognitiveExtractor
from cognimem.reconciler import MemoryReconciler
from cognimem.graph import KnowledgeGraph


class CogniMemory:
    """
    Central Cognitive Memory Orchestrator.
    Manages the full lifecycle: Extraction -> Semantic Reconciliation -> Vector Indexing ->
    Entity Knowledge Graph -> Context-Aware Multi-Factor Retrieval (ACT-R).
    """

    def __init__(
        self,
        storage: Optional[BaseVectorStore] = None,
        embeddings: Optional[BaseEmbeddings] = None,
        extractor: Optional[CognitiveExtractor] = None,
        reconciler: Optional[MemoryReconciler] = None,
        graph: Optional[KnowledgeGraph] = None,
        db_path: str = "cognimem.db",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None
    ):
        self.embeddings = embeddings or get_embedding_provider("openai", api_key=api_key)
        self.storage = storage or SQLiteVectorStore(db_path=db_path)
        self.extractor = extractor or CognitiveExtractor(model=model, api_key=api_key)
        self.reconciler = reconciler or MemoryReconciler(model=model, api_key=api_key)
        self.graph = graph or KnowledgeGraph(db_path=db_path)

    def add(
        self,
        messages_or_text: Union[str, List[Dict[str, str]]],
        user_id: str,
        session_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract facts from input, reconcile against existing memories in the vector store,
        and perform dynamic ADD/UPDATE/DELETE/NOOP operations.
        """
        metadata = metadata or {}
        extracted_facts = self.extractor.extract(messages_or_text)
        
        results: Dict[str, Any] = {
            "extracted_count": len(extracted_facts),
            "added": [],
            "updated": [],
            "deleted": [],
            "ignored": []
        }

        for fact_obj in extracted_facts:
            # 1. Embed the candidate fact
            fact_vector = self.embeddings.embed_text(fact_obj.fact)

            # 2. Search for semantically similar existing active memories for this user
            existing_matches = self.storage.search(
                query_vector=fact_vector,
                user_id=user_id,
                limit=3,
                threshold=0.25,
                status=MemoryStatus.ACTIVE
            )

            # Hybrid categorical matching: if fact has a known category (e.g. location, tech_stack),
            # link existing memories in the same category as reconciliation candidates
            if fact_obj.category and fact_obj.category != "general":
                all_active = self.storage.get_all(user_id=user_id, status=MemoryStatus.ACTIVE)
                for act in all_active:
                    if act.metadata.get("category") == fact_obj.category:
                        if not any(act.id == m[0].id for m in existing_matches):
                            existing_matches.append((act, 0.80))

            # 3. Determine reconciliation action (Conflict resolution & deduplication)
            decision = self.reconciler.reconcile(fact_obj, existing_matches)

            if decision.action == ReconciliationAction.ADD:
                item_id = str(uuid.uuid4())
                new_item = MemoryItem(
                    id=item_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    content=decision.resolved_fact or fact_obj.fact,
                    memory_type=fact_obj.memory_type,
                    confidence=fact_obj.confidence,
                    importance=fact_obj.importance,
                    metadata={**metadata, "category": fact_obj.category},
                    embedding=fact_vector
                )
                self.storage.upsert(new_item)
                for rel in getattr(fact_obj, "relations", []):
                    self.graph.add_relation(rel.subject, rel.predicate, rel.object, user_id=user_id, confidence=rel.confidence)
                results["added"].append({"id": item_id, "fact": new_item.content})

            elif decision.action == ReconciliationAction.UPDATE:
                new_id = str(uuid.uuid4())
                resolved_content = decision.resolved_fact or fact_obj.fact
                new_vector = self.embeddings.embed_text(resolved_content)

                # Mark older memory as superseded
                target_id = decision.target_memory_id or (existing_matches[0][0].id if existing_matches else None)
                if target_id:
                    self.storage.update_status(target_id, MemoryStatus.SUPERSEDED, superseded_by=new_id)

                # Insert updated memory
                new_item = MemoryItem(
                    id=new_id,
                    user_id=user_id,
                    agent_id=agent_id,
                    session_id=session_id,
                    content=resolved_content,
                    memory_type=fact_obj.memory_type,
                    confidence=fact_obj.confidence,
                    importance=fact_obj.importance,
                    metadata={**metadata, "category": fact_obj.category, "updated_from": target_id},
                    embedding=new_vector
                )
                self.storage.upsert(new_item)
                for rel in getattr(fact_obj, "relations", []):
                    self.graph.add_relation(rel.subject, rel.predicate, rel.object, user_id=user_id, confidence=rel.confidence)
                results["updated"].append({"id": new_id, "superseded_id": target_id, "fact": new_item.content})

            elif decision.action == ReconciliationAction.DELETE:
                target_id = decision.target_memory_id or (existing_matches[0][0].id if existing_matches else None)
                if target_id:
                    self.storage.update_status(target_id, MemoryStatus.ARCHIVED)
                    results["deleted"].append({"id": target_id, "rationale": decision.rationale})

            elif decision.action == ReconciliationAction.NOOP:
                results["ignored"].append({"fact": fact_obj.fact, "rationale": decision.rationale})

        return results

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        recency_weight: float = 0.15,
        similarity_threshold: float = 0.05,
        include_graph: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Context-aware semantic memory retrieval using the Stanford ACT-R Cognitive Scoring model:
        Score = w_sim * CosineSim + w_rec * RecencyDecay + w_imp * Importance + w_freq * FrequencyFactor
        """
        query_vector = self.embeddings.embed_text(query)
        matches = self.storage.search(
            query_vector=query_vector,
            user_id=user_id,
            limit=limit * 2,  # Fetch slightly wider pool for re-ranking
            threshold=similarity_threshold,
            status=MemoryStatus.ACTIVE
        )

        now = datetime.now(timezone.utc)
        ranked: List[Dict[str, Any]] = []

        # Cognitive weight distribution
        w_sim = 0.55 if recency_weight == 0.15 else max(0.2, 1.0 - recency_weight)
        w_rec = recency_weight
        w_imp = 0.20
        w_freq = 0.10
        total_w = w_sim + w_rec + w_imp + w_freq
        w_sim, w_rec, w_imp, w_freq = w_sim / total_w, w_rec / total_w, w_imp / total_w, w_freq / total_w

        for item, sim in matches:
            # 1. Recency decay (30-day half-life)
            delta_days = (now - item.created_at).total_seconds() / 86400.0
            recency_decay = math.exp(-0.023 * max(0.0, delta_days))

            # 2. Intrinsic importance
            importance = getattr(item, "importance", 0.5)

            # 3. Access frequency factor
            frequency_factor = min(1.0, (item.access_count + 1) / 10.0)

            # Composite cognitive score
            final_score = (
                w_sim * sim
                + w_rec * recency_decay
                + w_imp * importance
                + w_freq * frequency_factor
            )

            # Update access telemetry
            item.last_accessed_at = now
            item.access_count += 1
            self.storage.upsert(item)

            ranked.append({
                "id": item.id,
                "memory": item.content,
                "type": item.memory_type if isinstance(item.memory_type, str) else item.memory_type.value,
                "similarity": round(sim, 4),
                "importance": round(importance, 2),
                "access_count": item.access_count,
                "score": round(final_score, 4),
                "created_at": item.created_at.isoformat(),
                "metadata": item.metadata
            })

        # Sort descending by composite score
        ranked.sort(key=lambda x: x["score"], reverse=True)
        results = ranked[:limit]

        # Relational Knowledge Graph augmentation
        if include_graph:
            words = query.split()
            for w in words:
                if len(w) >= 4:
                    graph_edges = self.graph.get_connected_entities(w, user_id=user_id, max_hops=1)
                    for edge in graph_edges:
                        rel_str = f"Knowledge Graph Relation: ({edge['source']}) -> [{edge['relation']}] -> ({edge['target']})"
                        if not any(r["memory"] == rel_str for r in results):
                            results.append({
                                "id": f"graph-{edge['source']}-{edge['target']}",
                                "memory": rel_str,
                                "type": "graph_relation",
                                "similarity": 1.0,
                                "importance": 0.8,
                                "access_count": 1,
                                "score": 0.95,
                                "created_at": now.isoformat(),
                                "metadata": {"graph_edge": edge}
                            })

        return results[:limit]

    def get_all(self, user_id: str, status: MemoryStatus = MemoryStatus.ACTIVE) -> List[MemoryItem]:
        """Fetch all memories for a user."""
        return self.storage.get_all(user_id=user_id, status=status)

    def get_persona_card(self, user_id: str) -> str:
        """
        Synthesizes a structured Executive Persona Card for prompt injection and UI dashboards.
        """
        profile = self.get_profile(user_id)
        relations = self.graph.get_relations(user_id)

        card_lines = [f"### COGNITIVE PERSONA: {user_id.upper()}"]
        if profile["identity"]:
            card_lines.append("\n**Identity & Background**:")
            for item in profile["identity"]:
                card_lines.append(f"- {item}")
        if profile["constraints"]:
            card_lines.append("\n**Non-Negotiable Constraints & Allergies**:")
            for item in profile["constraints"]:
                card_lines.append(f"- {item}")
        if profile["preferences"]:
            card_lines.append("\n**Active Preferences & Tech Stack**:")
            for item in profile["preferences"]:
                card_lines.append(f"- {item}")
        if profile.get("other"):
            card_lines.append("\n**Other Context & Recorded Facts**:")
            for item in profile["other"]:
                card_lines.append(f"- {item}")
        if relations:
            card_lines.append("\n**Connected Knowledge Graph Triples**:")
            for r in relations[:6]:
                card_lines.append(f"- ({r.subject}) --[{r.predicate}]--> ({r.object})")

        if len(card_lines) == 1:
            card_lines.append("\n*(No active facts or traits recorded yet)*")

        return "\n".join(card_lines)

    def export_knowledge_graph(self, user_id: str) -> Dict[str, Any]:
        """Export graph elements for interactive visual rendering."""
        return self.graph.export_elements_for_ui(user_id)

    def get_profile(self, user_id: str) -> Dict[str, List[str]]:
        """Synthesize a structured user profile overview."""
        memories = self.storage.get_all(user_id=user_id, status=MemoryStatus.ACTIVE)
        profile: Dict[str, List[str]] = {
            "identity": [],
            "preferences": [],
            "constraints": [],
            "history": [],
            "other": []
        }

        for m in memories:
            m_type = m.memory_type.value if hasattr(m.memory_type, "value") else str(m.memory_type)
            cat = m.metadata.get("category", "")
            if m_type == "profile" or cat in ("identity", "profession", "location"):
                profile["identity"].append(m.content)
            elif cat in ("dietary", "constraint") or "allergic" in m.content.lower():
                profile["constraints"].append(m.content)
            elif m_type == "preference":
                profile["preferences"].append(m.content)
            elif m_type == "episodic":
                profile["history"].append(m.content)
            else:
                profile["other"].append(m.content)

        return profile

    def delete(self, memory_id: str) -> bool:
        """Permanently delete a memory record."""
        return self.storage.delete(memory_id)

    def reset(self, user_id: Optional[str] = None) -> int:
        """Clear vector memory and graph connections."""
        self.graph.clear(user_id)
        return self.storage.clear(user_id)
