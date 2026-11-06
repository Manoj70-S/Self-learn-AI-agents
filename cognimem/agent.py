from typing import List, Dict, Any, Optional
import os

from cognimem.memory import CogniMemory


DEFAULT_AGENT_SYSTEM_PROMPT = """You are an intelligent, proactive AI assistant equipped with an evolving long-term memory.
You remember user preferences, traits, constraints, and project history across separate sessions.

CRITICAL INSTRUCTIONS:
1. Always incorporate relevant long-term memories to tailor your advice, recommendations, and code snippets.
2. If the user previously specified preferences or restrictions (e.g. tech stack, food allergies, style preferences), seamlessly honor them without saying "According to my database".
3. Speak naturally, authoritatively, and concisely.

USER LONG-TERM MEMORY CONTEXT:
{memory_context}
"""


class CognitiveAgent:
    """
    Self-learning Conversational Agent equipped with dynamic long-term memory retrieval and updating.
    """

    def __init__(
        self,
        memory: Optional[CogniMemory] = None,
        system_prompt_template: str = DEFAULT_AGENT_SYSTEM_PROMPT,
        model: str = "gpt-4o",
        api_key: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.memory = memory or CogniMemory(api_key=self.api_key)
        self.system_prompt_template = system_prompt_template
        self.model = model
        self._openai_client = None
        if self.api_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=self.api_key)
            except ImportError:
                pass

    def chat(
        self,
        user_message: str,
        user_id: str,
        session_id: Optional[str] = None,
        short_term_history: Optional[List[Dict[str, str]]] = None,
        consolidate_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Execute an agent turn:
        1. Query long-term memory for semantic context.
        2. Inject memory into prompt.
        3. Generate response.
        4. Consolidate new conversation turns into long-term memory.
        """
        short_term_history = short_term_history or []

        # Step 1: Semantic memory search
        retrieved_memories = self.memory.search(
            query=user_message,
            user_id=user_id,
            limit=5,
            recency_weight=0.15
        )

        facts_text = "\n".join([f"- {m['memory']}" for m in retrieved_memories])
        memory_context = facts_text if facts_text else "No relevant long-term memories found."

        # Step 2: Build messages payload
        system_prompt = self.system_prompt_template.format(memory_context=memory_context)
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(short_term_history)
        messages.append({"role": "user", "content": user_message})

        # Step 3: Generate response
        if self._openai_client:
            response = self._openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            assistant_reply = response.choices[0].message.content
        else:
            # Deterministic intelligent responder for offline testing / zero-key mode
            assistant_reply = self._generate_simulated_reply(user_message, retrieved_memories)

        # Step 4: Background memory consolidation
        consolidation_report = None
        if consolidate_memory:
            turn = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_reply}
            ]
            consolidation_report = self.memory.add(
                messages_or_text=turn,
                user_id=user_id,
                session_id=session_id
            )

        return {
            "reply": assistant_reply,
            "retrieved_memories": retrieved_memories,
            "consolidation": consolidation_report
        }

    def _generate_simulated_reply(self, user_message: str, retrieved_memories: List[Dict[str, Any]]) -> str:
        """Fallback response generator acknowledging retrieved memory context."""
        mem_snippets = [m["memory"] for m in retrieved_memories]
        msg_lower = user_message.lower()

        if "thai" in msg_lower or "dinner" in msg_lower or "food" in msg_lower:
            reply = "I'd suggest a fragrant Thai Green Curry with jasmine rice. "
            if any("allergic" in s.lower() for s in mem_snippets):
                reply += "I've made sure to omit peanuts, fish sauce, and shellfish to strictly respect your allergy requirements!"
            return reply
        elif "backend" in msg_lower or "framework" in msg_lower or "recommend" in msg_lower:
            reply = "For your architecture, "
            if any("fastapi" in s.lower() for s in mem_snippets):
                reply += "I recommend structuring your service with FastAPI and PostgreSQL, keeping your preferred async stack clean and modular."
            else:
                reply += "FastAPI with async SQLAlchemy and PostgreSQL provides an excellent, high-performance foundation."
            return reply
        else:
            if mem_snippets:
                recalled = "; ".join(mem_snippets[:2])
                return f"Understood! Keeping in mind that {recalled}, I can assist you with your request: '{user_message}'."
            return f"Hello! How can I assist you with your project today regarding '{user_message}'?"
