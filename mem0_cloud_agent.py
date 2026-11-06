"""
Interactive Mem0 Cloud Conversational Agent
Powered by your Mem0 Cloud Platform API key.
Supports interactive chat, real-time memory inspection, semantic search, and multi-user switching.
"""

import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

from mem0 import MemoryClient

# Optional OpenAI client for rich conversational replies
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        pass


class Mem0CloudAgent:
    def __init__(self, api_key: str, default_user_id: str = "user_demo"):
        self.api_key = api_key
        self.client = MemoryClient(api_key=api_key)
        self.user_id = default_user_id

    def switch_user(self, new_user_id: str):
        self.user_id = new_user_id
        print(f"\n[System] Switched active user to: '{self.user_id}'")

    def show_memories(self):
        print(f"\n[Mem0 Cloud] Fetching all active memories for '{self.user_id}'...")
        try:
            res = self.client.get_all(filters={"user_id": self.user_id})
            memories = res.get("results", [])
            if not memories:
                print("   No memories found for this user on Mem0 Cloud.")
                return
            for idx, m in enumerate(memories, 1):
                mem_text = m.get("memory", "")
                mem_id = m.get("id", "")
                created = m.get("created_at", "")[:19]
                print(f"   {idx}. {mem_text}")
                print(f"      (ID: {mem_id} | Created: {created})")
        except Exception as e:
            print(f"   Error retrieving memories: {e}")

    def search_memories(self, query: str):
        print(f"\n[Mem0 Cloud] Semantic search for: '{query}' (User: '{self.user_id}')...")
        try:
            res = self.client.search(query, filters={"user_id": self.user_id})
            memories = res.get("results", [])
            if not memories:
                print("   No matching memories found.")
                return
            for idx, m in enumerate(memories, 1):
                print(f"   {idx}. {m.get('memory')} (Score: {m.get('score', 'N/A')})")
        except Exception as e:
            print(f"   Error searching memories: {e}")

    def clear_memories(self):
        confirm = input(f"Are you sure you want to wipe all memories for '{self.user_id}' on Mem0 Cloud? (y/N): ").strip().lower()
        if confirm == 'y':
            try:
                self.client.delete_all(filters={"user_id": self.user_id})
                print(f"[Mem0 Cloud] All memories wiped for '{self.user_id}'.")
            except Exception as e:
                print(f"[Mem0 Cloud] Error deleting memories: {e}")

    def chat_turn(self, user_message: str) -> str:
        # Step 1: Query Mem0 Cloud for relevant context
        try:
            search_res = self.client.search(user_message, filters={"user_id": self.user_id})
            retrieved = [m.get("memory") for m in search_res.get("results", []) if m.get("memory")]
        except Exception:
            retrieved = []

        context_str = ""
        if retrieved:
            context_str = "\n".join([f"- {fact}" for fact in retrieved])
        else:
            context_str = "No specific past context recalled."

        # Step 2: Formulate Assistant Response
        if openai_client:
            system_prompt = f"""You are a personalized, context-aware AI assistant.
You possess a long-term memory of the user provided by Mem0 Cloud.
Incorporate recalled facts naturally into your answers without explicitly saying 'According to Mem0'.

Recalled User Memory Context:
{context_str}
"""
            try:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.7
                )
                assistant_reply = resp.choices[0].message.content
            except Exception as e:
                assistant_reply = f"(OpenAI Error: {e}) Understood: '{user_message}'."
        else:
            # Smart conversational simulator when OpenAI API key is omitted
            if retrieved:
                top_context = retrieved[0]
                assistant_reply = (
                    f"I hear you! Keeping in mind that {top_context}, "
                    f"I've tailored my response to your request: '{user_message}'."
                )
            else:
                assistant_reply = f"Got it! I have noted that for your profile: '{user_message}'."

        # Step 3: Record conversation to Mem0 Cloud
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_reply}
        ]
        try:
            self.client.add(messages, user_id=self.user_id)
        except Exception as e:
            print(f"[Warning: Mem0 Cloud background sync error: {e}]")

        return assistant_reply, retrieved


def print_banner(user_id: str):
    print("=" * 70)
    print("      MEM0 CLOUD CONVERSATIONAL AGENT")
    print("=" * 70)
    print(f"Active User ID: {user_id}")
    print("Commands:")
    print("  /memories        - Show all active memories for current user")
    print("  /search <query>  - Run semantic vector search on Mem0 Cloud")
    print("  /switch <id>     - Switch to another user ID")
    print("  /reset           - Delete all memories for current user")
    print("  /exit            - Quit chat session")
    print("=" * 70 + "\n")


def main():
    api_key = os.getenv("MEM0_API_KEY")
    if not api_key:
        print("ERROR: MEM0_API_KEY is not set in .env")
        sys.exit(1)

    agent = Mem0CloudAgent(api_key=api_key, default_user_id="alex_cloud_user")
    print_banner(agent.user_id)

    # If run in non-interactive / piped mode, execute a demonstration flow
    if not sys.stdin.isatty():
        demo_queries = [
            "What do you know about my tech stack and preferences?",
            "Can you suggest a healthy dinner option?",
            "/memories"
        ]
        for q in demo_queries:
            print(f"\nUser > {q}")
            if q.startswith("/memories"):
                agent.show_memories()
            elif q.startswith("/search "):
                agent.search_memories(q[8:])
            else:
                reply, recalled = agent.chat_turn(q)
                if recalled:
                    print(f"[Mem0 Context Recalled]: {recalled}")
                print(f"Agent > {reply}")
        return

    # Interactive REPL mode
    while True:
        try:
            user_input = input(f"\n[{agent.user_id}] > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                print("Goodbye!")
                break
            elif user_input.lower() == "/memories":
                agent.show_memories()
            elif user_input.startswith("/search "):
                agent.search_memories(user_input[8:].strip())
            elif user_input.startswith("/switch "):
                agent.switch_user(user_input[8:].strip())
            elif user_input.lower() == "/reset":
                agent.clear_memories()
            else:
                reply, recalled = agent.chat_turn(user_input)
                if recalled:
                    print(f"\n[Mem0 Recalled]:")
                    for r in recalled:
                        print(f"  * {r}")
                print(f"\nAgent > {reply}")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    main()
