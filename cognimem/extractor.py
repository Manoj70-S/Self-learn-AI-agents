from typing import List, Dict, Any, Optional, Union
import json
import re
import os

from cognimem.models import ExtractedFact, MemoryType, EntityRelation


EXTRACTION_SYSTEM_PROMPT = """You are an advanced Cognitive Memory & Knowledge Graph Extraction module for an AI assistant.
Your job is to analyze conversations and extract salient, long-term atomic facts AND entity relationship triples.

Strict Guidelines:
1. Extract ONLY facts that have lasting relevance for future conversations (identity, preferences, tech choices, constraints).
2. Ignore ephemeral chit-chat, greetings, polite fillers, and fleeting task instructions.
3. Normalize each fact into a concise, third-person declarative sentence starting with "User..." (e.g. "User prefers FastAPI").
4. Rate importance from 0.0 (trivial) to 1.0 (vital, e.g. severe allergies=0.95, core tech=0.85, names=0.90).
5. Extract entity-relation triples (subject, predicate, object) for the knowledge graph (e.g. "User", "PREFERS", "FastAPI").

Return your response strictly as a JSON object matching this schema:
{
  "facts": [
    {
      "fact": "User prefers FastAPI for backend development.",
      "memory_type": "preference",
      "category": "tech_stack",
      "confidence": 0.95,
      "importance": 0.85,
      "temporal_context": "current",
      "relations": [
        {"subject": "User", "predicate": "PREFERS_FRAMEWORK", "object": "FastAPI"}
      ]
    }
  ]
}
If no lasting facts are found, return: {"facts": []}
"""


class CognitiveExtractor:
    """Extracts atomic, normalized facts from conversations."""

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

    def extract(self, input_data: Union[str, List[Dict[str, str]]]) -> List[ExtractedFact]:
        """
        Extract facts from either a raw text string or a conversation list:
        [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        """
        # Format text to analyze
        if isinstance(input_data, str):
            dialogue_text = f"User: {input_data.strip()}"
        elif isinstance(input_data, list):
            turns = []
            for m in input_data:
                role = m.get("role", "user").capitalize()
                content = m.get("content", "")
                turns.append(f"{role}: {content}")
            dialogue_text = "\n".join(turns)
        else:
            return []

        if not dialogue_text.strip():
            return []

        # If OpenAI client is available, run high-precision LLM extraction
        if self._openai_client:
            try:
                return self._extract_with_llm(dialogue_text)
            except Exception as e:
                # Log and fallback gracefully
                print(f"[CogniMem Extractor] LLM extraction error: {e}. Falling back to rule-based parser.")

        return self._extract_with_rules(dialogue_text)

    def _extract_with_llm(self, dialogue_text: str) -> List[ExtractedFact]:
        response = self._openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract memory facts from this dialogue:\n\n{dialogue_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        facts_raw = data.get("facts", [])
        results: List[ExtractedFact] = []
        for f in facts_raw:
            try:
                m_type = f.get("memory_type", "preference").lower()
                rels = [
                    EntityRelation(
                        subject=r.get("subject", "User"),
                        predicate=r.get("predicate", "RELATED_TO").upper(),
                        object=r.get("object", "")
                    )
                    for r in f.get("relations", [])
                    if r.get("object")
                ]
                results.append(ExtractedFact(
                    fact=f["fact"],
                    memory_type=MemoryType(m_type) if m_type in [e.value for e in MemoryType] else MemoryType.PREFERENCE,
                    category=f.get("category", "general"),
                    confidence=float(f.get("confidence", 1.0)),
                    importance=float(f.get("importance", 0.5)),
                    temporal_context=f.get("temporal_context"),
                    relations=rels
                ))
            except Exception:
                continue
        return results

    def _extract_with_rules(self, dialogue_text: str) -> List[ExtractedFact]:
        """
        Deterministic, rule-based extraction fallback for offline/testing scenarios.
        Extracts patterns such as 'I prefer X', 'I am allergic to X', 'I live in X', etc.
        """
        patterns = [
            (r"(?:i prefer|i like|i love)\s+(.+?)(?:\.|$|,|and)", MemoryType.PREFERENCE, "preference", "User prefers {}", 0.75, "PREFERS"),
            (r"(?:i hate|i dislike|i avoid)\s+(.+?)(?:\.|$|,|and)", MemoryType.PREFERENCE, "preference", "User dislikes {}", 0.70, "DISLIKES"),
            (r"(?:i am(?:\s+\w+)?\s+allergic to|i have an allergy to)\s+(.+?)(?:\.|$|,|and)", MemoryType.PROFILE, "dietary", "User is allergic to {}", 0.95, "ALLERGIC_TO"),
            (r"(?:i live in|i am based in|i moved to)\s+(.+?)(?:\.|$|,|and)", MemoryType.PROFILE, "location", "User lives in {}", 0.85, "LIVES_IN"),
            (r"(?:my name is|i am called|call me)\s+([a-zA-Z]+)", MemoryType.PROFILE, "identity", "User's name is {}", 0.90, "NAME_IS"),
            (r"(?:i work as a|i am a)\s+([a-zA-Z\s]+?)(?:developer|engineer|scientist|designer|manager)", MemoryType.PROFILE, "profession", "User works as {} developer/engineer", 0.80, "WORKS_AS"),
            (r"(?:i use|i work with)\s+(python|fastapi|django|react|vue|svelte|docker|kubernetes|qdrant|postgresql)", MemoryType.PREFERENCE, "tech_stack", "User uses {}", 0.80, "USES_TECH"),
            (r"(?:i switched from)\s+(.+?)\s+to\s+(.+?)(?:\.|$|,|for)", MemoryType.PREFERENCE, "tech_stack", "User prefers {} (switched from {})", 0.85, "PREFERS"),
        ]

        results: List[ExtractedFact] = []
        lower_text = dialogue_text.lower()

        for pattern, mem_type, cat, template, imp, rel_pred in patterns:
            matches = re.finditer(pattern, lower_text)
            for m in matches:
                groups = m.groups()
                relations = []
                if len(groups) == 1:
                    val = groups[0].strip().capitalize()
                    statement = template.format(val)
                    relations.append(EntityRelation(subject="User", predicate=rel_pred, object=val))
                elif len(groups) == 2:
                    val1 = groups[0].strip().capitalize()
                    val2 = groups[1].strip().capitalize()
                    statement = template.format(val2, val1) if "(switched from {})" in template else template.format(val1, val2)
                    relations.append(EntityRelation(subject="User", predicate=rel_pred, object=val2 if "(switched from {})" in template else val1))
                else:
                    continue

                results.append(ExtractedFact(
                    fact=statement,
                    memory_type=mem_type,
                    category=cat,
                    confidence=0.9,
                    importance=imp,
                    relations=relations
                ))

        # If input was a direct sentence and no pattern matched, only extract if it is a statement, not a question
        if not results:
            clean_lines = [line.replace("User:", "").strip() for line in dialogue_text.splitlines() if line.startswith("User:")]
            for cl in clean_lines:
                cl_clean = cl.strip()
                # Ignore questions and conversational greetings
                if cl_clean.endswith("?") or any(cl_clean.lower().startswith(q) for q in ["what", "can you", "how", "why", "where", "is there", "recommend", "hi", "hello", "thanks"]):
                    continue
                if len(cl_clean.split()) >= 4:
                    results.append(ExtractedFact(
                        fact=f"User mentioned: {cl_clean}",
                        memory_type=MemoryType.SEMANTIC,
                        category="general",
                        confidence=0.75
                    ))

        return results
