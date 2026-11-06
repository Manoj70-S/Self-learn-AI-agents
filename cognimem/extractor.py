from typing import List, Dict, Any, Optional, Union
import json
import re
import os

from cognimem.models import ExtractedFact, MemoryType


EXTRACTION_SYSTEM_PROMPT = """You are an advanced Cognitive Memory Extraction module for an AI assistant.
Your job is to analyze conversations and extract salient, long-term atomic facts, user preferences, technical constraints, habits, and biographical details.

Strict Guidelines:
1. Extract ONLY facts that have lasting relevance for future conversations (e.g., identity, preferences, stack choices, dietary restrictions, recurring goals).
2. Ignore ephemeral chit-chat, greetings, polite fillers, and fleeting task instructions (e.g., "summarize this file", "hello", "thank you").
3. Normalize each fact into a concise, third-person declarative sentence starting with "User..." (e.g., "User prefers PostgreSQL over MongoDB", "User is strictly allergic to peanuts").
4. Assign an appropriate memory_type:
   - "profile": Immutable identity or core life attributes (name, home location, profession).
   - "preference": Tools, flavors, coding styles, language choices, favorite frameworks.
   - "episodic": Notable past events, project migrations, or milestones.
   - "semantic": Domain knowledge, project requirements, architectural decisions.
5. Provide confidence (0.0 to 1.0).

Return your response strictly as a JSON object matching this schema:
{
  "facts": [
    {
      "fact": "User prefers FastAPI for backend development.",
      "memory_type": "preference",
      "category": "tech_stack",
      "confidence": 0.95,
      "temporal_context": "current"
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
                results.append(ExtractedFact(
                    fact=f["fact"],
                    memory_type=MemoryType(m_type) if m_type in [e.value for e in MemoryType] else MemoryType.PREFERENCE,
                    category=f.get("category", "general"),
                    confidence=float(f.get("confidence", 1.0)),
                    temporal_context=f.get("temporal_context")
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
            (r"(?:i prefer|i like|i love)\s+(.+?)(?:\.|$|,|and)", MemoryType.PREFERENCE, "preference", "User prefers {}"),
            (r"(?:i hate|i dislike|i avoid)\s+(.+?)(?:\.|$|,|and)", MemoryType.PREFERENCE, "preference", "User dislikes {}"),
            (r"(?:i am(?:\s+\w+)?\s+allergic to|i have an allergy to)\s+(.+?)(?:\.|$|,|and)", MemoryType.PROFILE, "dietary", "User is allergic to {}"),
            (r"(?:i live in|i am based in|i moved to)\s+(.+?)(?:\.|$|,|and)", MemoryType.PROFILE, "location", "User lives in {}"),
            (r"(?:my name is|i am called|call me)\s+([A-Z][a-z]+)", MemoryType.PROFILE, "identity", "User's name is {}"),
            (r"(?:i work as a|i am a)\s+([a-zA-Z\s]+?)(?:developer|engineer|scientist|designer|manager)", MemoryType.PROFILE, "profession", "User works as {} developer/engineer"),
            (r"(?:i use|i work with)\s+(python|fastapi|django|react|vue|svelte|docker|kubernetes|qdrant|postgresql)", MemoryType.PREFERENCE, "tech_stack", "User uses {}"),
            (r"(?:i switched from)\s+(.+?)\s+to\s+(.+?)(?:\.|$|,|for)", MemoryType.PREFERENCE, "tech_stack", "User prefers {} (switched from {})"),
        ]

        results: List[ExtractedFact] = []
        lower_text = dialogue_text.lower()

        for pattern, mem_type, cat, template in patterns:
            matches = re.finditer(pattern, lower_text)
            for m in matches:
                groups = m.groups()
                if len(groups) == 1:
                    val = groups[0].strip()
                    statement = template.format(val.capitalize())
                elif len(groups) == 2:
                    val1 = groups[0].strip().capitalize()
                    val2 = groups[1].strip().capitalize()
                    # If template has 2 slots, format appropriately
                    statement = template.format(val2, val1) if "(switched from {})" in template else template.format(val1, val2)
                else:
                    continue

                results.append(ExtractedFact(
                    fact=statement,
                    memory_type=mem_type,
                    category=cat,
                    confidence=0.9
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
