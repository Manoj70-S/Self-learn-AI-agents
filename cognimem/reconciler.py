from typing import List, Optional, Tuple
import json
import os
import re

from cognimem.models import (
    MemoryItem,
    ExtractedFact,
    ReconciliationAction,
    ReconciliationDecision,
)


RECONCILIATION_SYSTEM_PROMPT = """You are a Memory Consolidation and Conflict Resolution engine for an AI agent.
Your task is to compare a NEW candidate fact against EXISTING memories stored for a user, and decide the appropriate action.

Possible actions:
1. "add": The new fact contains genuinely new information not covered by existing memories.
2. "update": The new fact contradicts, supersedes, or meaningfully updates an existing memory (e.g. "User lives in Amsterdam" -> "User moved to Berlin"). Specify target_memory_id.
3. "delete": The new fact explicitly revokes an old memory (e.g. "User no longer drinks coffee"). Specify target_memory_id.
4. "noop": The new fact is already known, semantically identical, or redundant. No change needed.

Respond strictly with a JSON object:
{
  "action": "add" | "update" | "delete" | "noop",
  "target_memory_id": "string or null",
  "resolved_fact": "clean third-person statement or null",
  "rationale": "Brief reason for decision"
}
"""


class MemoryReconciler:
    """Performs semantic deduplication, contradiction detection, and lifecycle actions."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._openai_client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.api_key)
            except ImportError:
                pass

    def reconcile(
        self,
        candidate: ExtractedFact,
        existing_matches: List[Tuple[MemoryItem, float]],
        similarity_threshold: float = 0.35
    ) -> ReconciliationDecision:
        """
        Compare candidate fact with retrieved existing memories (sorted by similarity).
        Returns a ReconciliationDecision.
        """
        if not existing_matches:
            return ReconciliationDecision(
                action=ReconciliationAction.ADD,
                resolved_fact=candidate.fact,
                rationale="No existing memories found."
            )

        top_match, top_similarity = existing_matches[0]

        # If similarity is low and no shared slot/category, treat as novel ADD
        has_cat_match = (
            candidate.category
            and candidate.category != "general"
            and top_match.metadata.get("category") == candidate.category
        )
        if top_similarity < similarity_threshold and not has_cat_match:
            return ReconciliationDecision(
                action=ReconciliationAction.ADD,
                resolved_fact=candidate.fact,
                rationale="No semantically similar existing memory found."
            )

        # Use LLM for intelligent semantic contradiction detection if available
        if self._openai_client:
            try:
                return self._reconcile_with_llm(candidate, existing_matches)
            except Exception as e:
                print(f"[CogniMem Reconciler] LLM error: {e}. Using deterministic fallback.")

        # Deterministic / rule-based fallback
        return self._reconcile_heuristically(candidate, top_match, top_similarity)

    def _reconcile_with_llm(
        self,
        candidate: ExtractedFact,
        existing_matches: List[Tuple[MemoryItem, float]]
    ) -> ReconciliationDecision:
        memories_text = "\n".join([
            f"- [ID: {item.id}] (Similarity: {sim:.2f}) {item.content}"
            for item, sim in existing_matches[:3]
        ])

        user_prompt = f"""NEW FACT TO EVALUATE:
"{candidate.fact}" (Category: {candidate.category}, Type: {candidate.memory_type})

EXISTING MEMORIES:
{memories_text}
"""

        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RECONCILIATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        data = json.loads(response.choices[0].message.content)
        action_str = data.get("action", "add").lower()
        action = ReconciliationAction(action_str) if action_str in [a.value for a in ReconciliationAction] else ReconciliationAction.ADD

        return ReconciliationDecision(
            action=action,
            target_memory_id=data.get("target_memory_id"),
            resolved_fact=data.get("resolved_fact") or candidate.fact,
            rationale=data.get("rationale", "")
        )

    def _reconcile_heuristically(
        self,
        candidate: ExtractedFact,
        top_match: MemoryItem,
        similarity: float
    ) -> ReconciliationDecision:
        cand_lower = candidate.fact.lower().strip()
        top_lower = top_match.content.lower().strip()

        cand_words = set(re.findall(r"\w+", cand_lower))
        top_words = set(re.findall(r"\w+", top_lower))

        # 1. Exact match (identical semantic content) -> NOOP
        if cand_words == top_words:
            return ReconciliationDecision(
                action=ReconciliationAction.NOOP,
                target_memory_id=top_match.id,
                rationale="Fact already recorded with identical content."
            )

        # 2. Detect conflicting keywords in the same domain (e.g. location, stack)
        conflict_signals = ["moved to", "switched from", "no longer", "instead of", "stopped"]
        if any(sig in cand_lower for sig in conflict_signals):
            return ReconciliationDecision(
                action=ReconciliationAction.UPDATE,
                target_memory_id=top_match.id,
                resolved_fact=candidate.fact,
                rationale=f"Updated prior memory [{top_match.id}] based on change indicator."
            )

        # 3. Categorical single-slot attributes (e.g., location, profession, identity)
        top_cat = top_match.metadata.get("category", "")
        cand_cat = candidate.category or ""
        if (top_cat and cand_cat and top_cat == cand_cat and cand_cat in ("location", "profession", "identity")) or \
           ("lives in" in cand_lower and "lives in" in top_lower):
            return ReconciliationDecision(
                action=ReconciliationAction.UPDATE,
                target_memory_id=top_match.id,
                resolved_fact=candidate.fact,
                rationale=f"Updated prior {cand_cat or 'profile'} memory [{top_match.id}]."
            )

        # If similarity is very high (>0.75) and subject matter matches
        if similarity >= 0.75:
            # Check if it updates location, preferred tech, etc.
            if ("lives in" in cand_lower and "lives in" in top_lower) or \
               ("prefers" in cand_lower and "prefers" in top_lower):
                return ReconciliationDecision(
                    action=ReconciliationAction.UPDATE,
                    target_memory_id=top_match.id,
                    resolved_fact=candidate.fact,
                    rationale=f"Preference or profile updated over [{top_match.id}]."
                )

        return ReconciliationDecision(
            action=ReconciliationAction.ADD,
            resolved_fact=candidate.fact,
            rationale="Supplementary detail."
        )
