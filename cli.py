"""
CogniMem Terminal CLI: Pure Terminal-Native Cognitive Memory Interface.
No web servers or browser UIs - 100% terminal execution.

Usage:
  python cli.py                   # Starts interactive terminal chat
  python cli.py chat              # Starts interactive terminal chat
  python cli.py memories          # Inspects active & superseded memories
  python cli.py graph             # Inspects Knowledge Graph triples
  python cli.py persona           # Prints the synthesized Persona Card
  python cli.py search "query"    # Runs ACT-R multi-factor semantic search
  python cli.py add "statement"   # Directly indexes a fact into memory
  python cli.py reset             # Purges memories for a user
"""

import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

from cognimem import CogniMemory, CognitiveAgent, MemoryStatus
from cognimem.storage.sqlite_store import SQLiteVectorStore
from cognimem.embeddings import get_embedding_provider

# ANSI Color codes for clean terminal output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def get_system(provider: str = None):
    prov = (provider or os.getenv("COGNIMEM_STORAGE_PROVIDER", "sqlite")).lower()
    if prov == "qdrant" and os.getenv("QDRANT_URL"):
        try:
            from cognimem.storage.qdrant_store import QdrantVectorStore
            storage = QdrantVectorStore(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                collection_name=os.getenv("QDRANT_COLLECTION", "agent_memories")
            )
            storage_name = "Qdrant Cloud"
        except Exception as e:
            storage = SQLiteVectorStore(db_path="cognimem.db")
            storage_name = f"SQLite Vector Engine (Fallback: {e})"
    else:
        storage = SQLiteVectorStore(db_path="cognimem.db")
        storage_name = "SQLite Vector Engine (`cognimem.db`)"

    embeddings = get_embedding_provider("deterministic", dimensions=1536)
    memory = CogniMemory(storage=storage, embeddings=embeddings, db_path="cognimem.db")
    agent = CognitiveAgent(memory=memory)
    return memory, agent, storage_name


def print_banner(user_id: str, storage_name: str = "SQLite Vector Engine (`cognimem.db`)"):
    print(f"\n{CYAN}{BOLD}{'=' * 72}{RESET}")
    print(f"{CYAN}{BOLD}       🧠  COGNIMEM : TERMINAL COGNITIVE AGENT INTERFACE  🧠{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 72}{RESET}")
    print(f"{BOLD}Active User:{RESET} {GREEN}{user_id}{RESET}  |  {BOLD}Storage:{RESET} {storage_name}")
    print(f"{DIM}Commands: /memories | /graph | /persona | /search <query> | /switch <id> | /reset | /exit{RESET}")
    print(f"{CYAN}{'-' * 72}{RESET}\n")


def cmd_memories(memory: CogniMemory, user_id: str):
    active = memory.get_all(user_id, status=MemoryStatus.ACTIVE)
    superseded = memory.get_all(user_id, status=MemoryStatus.SUPERSEDED)

    print(f"\n{BOLD}{GREEN}=== ACTIVE MEMORIES ({len(active)}) ==={RESET}")
    if not active:
        print(f"{DIM}  (No active memories found for user '{user_id}'){RESET}")
    else:
        print(f"{DIM}{'ID':<10} {'IMP':<6} {'ACCESS':<8} {'CONTENT'}{RESET}")
        print(f"{DIM}{'-' * 72}{RESET}")
        for m in active:
            m_id = m.id[:8]
            imp = f"{getattr(m, 'importance', 0.5):.2f}"
            cnt = str(m.access_count)
            print(f"{CYAN}{m_id:<10}{RESET} {YELLOW}{imp:<6}{RESET} {cnt:<8} {m.content}")

    print(f"\n{BOLD}{RED}=== SUPERSEDED / AUDIT TRAIL ({len(superseded)}) ==={RESET}")
    if not superseded:
        print(f"{DIM}  (No superseded memories yet){RESET}")
    else:
        print(f"{DIM}{'ID':<10} {'REPLACED BY':<14} {'CONTENT'}{RESET}")
        print(f"{DIM}{'-' * 72}{RESET}")
        for s in superseded:
            s_id = s.id[:8]
            rep = (s.superseded_by[:8] + "...") if s.superseded_by else "None"
            print(f"{RED}{s_id:<10}{RESET} {DIM}{rep:<14}{RESET} {s.content}")
    print()


def cmd_graph(memory: CogniMemory, user_id: str):
    relations = memory.graph.get_relations(user_id)
    print(f"\n{BOLD}{MAGENTA}=== KNOWLEDGE GRAPH TRIPLES ({len(relations)}) ==={RESET}")
    if not relations:
        print(f"{DIM}  (Knowledge Graph is empty for user '{user_id}'){RESET}\n")
        return

    print(f"{DIM}{'SUBJECT':<16} {'PREDICATE / RELATION':<24} {'OBJECT':<20}{RESET}")
    print(f"{DIM}{'-' * 72}{RESET}")
    for r in relations:
        pred_str = f"--[{r.predicate}]-->"
        print(f"{CYAN}{r.subject:<16}{RESET} {YELLOW}{pred_str:<24}{RESET} {GREEN}{r.object:<20}{RESET}")
    print()


def cmd_persona(memory: CogniMemory, user_id: str):
    card = memory.get_persona_card(user_id)
    print(f"\n{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"{BOLD}{card}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}\n")


def cmd_search(memory: CogniMemory, user_id: str, query: str):
    results = memory.search(query=query, user_id=user_id, limit=5, similarity_threshold=0.0, include_graph=True)
    print(f"\n{BOLD}ACT-R Multi-Factor Search for:{RESET} '{YELLOW}{query}{RESET}' (User: {GREEN}{user_id}{RESET})\n")
    if not results:
        print(f"{DIM}  (No relevant memories found exceeding similarity threshold){RESET}\n")
        return

    print(f"{DIM}{'SCORE':<8} {'SIM':<8} {'IMP':<6} {'TYPE':<14} {'RECALLED MEMORY'}{RESET}")
    print(f"{DIM}{'-' * 72}{RESET}")
    for r in results:
        score = f"{r.get('score', 0):.3f}"
        sim = f"{r.get('similarity', 0):.2f}"
        imp = f"{r.get('importance', 0.5):.2f}"
        m_type = str(r.get('type', ''))[:13]
        print(f"{GREEN}{score:<8}{RESET} {CYAN}{sim:<8}{RESET} {YELLOW}{imp:<6}{RESET} {DIM}{m_type:<14}{RESET} {r.get('memory')}")
    print()


def interactive_chat(memory: CogniMemory, agent: CognitiveAgent, default_user: str = "alex_engineer", storage_name: str = "SQLite Vector Engine (`cognimem.db`)"):
    user_id = default_user
    print_banner(user_id, storage_name)

    while True:
        try:
            prompt = input(f"{BOLD}[{user_id}] > {RESET}").strip()
            if not prompt:
                continue

            if prompt.lower() in ("/exit", "/quit", "exit", "quit"):
                print(f"{DIM}Session ended. Goodbye!{RESET}\n")
                break
            elif prompt.lower() == "/memories":
                cmd_memories(memory, user_id)
            elif prompt.lower() == "/graph":
                cmd_graph(memory, user_id)
            elif prompt.lower() == "/persona":
                cmd_persona(memory, user_id)
            elif prompt.startswith("/search "):
                cmd_search(memory, user_id, prompt[8:].strip())
            elif prompt.startswith("/switch "):
                new_user = prompt[8:].strip()
                if new_user:
                    user_id = new_user
                    print(f"{GREEN}Switched active user to: '{user_id}'{RESET}\n")
            elif prompt.lower() == "/reset":
                confirm = input(f"{RED}Purge all memories for '{user_id}'? (y/N): {RESET}").strip().lower()
                if confirm == "y":
                    memory.reset(user_id)
                    print(f"{GREEN}Memories and graph wiped for '{user_id}'.{RESET}\n")
            elif prompt.lower() == "/help":
                print(f"""
{BOLD}CogniMem Terminal Commands:{RESET}
  /memories        - Show table of active facts and superseded audit trail
  /graph           - Display Knowledge Graph entity-relation triples
  /persona         - Print the synthesized user persona card
  /search <query>  - Run ACT-R cognitive multi-factor semantic search
  /switch <id>     - Switch to another user ID (multi-tenant testing)
  /reset           - Purge memory for current user
  /exit            - Quit chat
""")
            else:
                # Normal conversational turn
                res = agent.chat(user_message=prompt, user_id=user_id)
                recalled = res.get("retrieved_memories", [])
                
                if recalled:
                    print(f"\n{DIM}⚡ Recalled Context ({len(recalled)} facts):{RESET}")
                    for r in recalled[:3]:
                        print(f"  {CYAN}* [{r.get('score', 0):.2f}]{RESET} {r.get('memory')}")
                    print()

                print(f"{BOLD}{GREEN}Agent >{RESET} {res['reply']}\n")

        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Session closed.{RESET}")
            break


def main():
    parser = argparse.ArgumentParser(description="CogniMem Terminal Cognitive Memory CLI")
    parser.add_argument("--storage", choices=["sqlite", "qdrant"], default=None, help="Vector storage engine (default: sqlite or env)")
    subparsers = parser.add_subparsers(dest="command")

    # Command: chat
    p_chat = subparsers.add_parser("chat", help="Start interactive terminal chat session")
    p_chat.add_argument("--user", default="alex_engineer", help="Target User ID")

    # Command: memories
    p_mem = subparsers.add_parser("memories", help="List active and superseded memories")
    p_mem.add_argument("--user", default="alex_engineer", help="Target User ID")

    # Command: graph
    p_graph = subparsers.add_parser("graph", help="Show knowledge graph relational triples")
    p_graph.add_argument("--user", default="alex_engineer", help="Target User ID")

    # Command: persona
    p_pers = subparsers.add_parser("persona", help="Print synthesized Persona Card")
    p_pers.add_argument("--user", default="alex_engineer", help="Target User ID")

    # Command: search
    p_search = subparsers.add_parser("search", help="Run ACT-R cognitive semantic search")
    p_search.add_argument("query", help="Query string to search")
    p_search.add_argument("--user", default="alex_engineer", help="Target User ID")

    # Command: add
    p_add = subparsers.add_parser("add", help="Directly index a declarative statement")
    p_add.add_argument("statement", help="Statement text to add")
    p_add.add_argument("--user", default="alex_engineer", help="Target User ID")

    # Command: reset
    p_reset = subparsers.add_parser("reset", help="Wipe all memories for a user")
    p_reset.add_argument("--user", default="alex_engineer", help="Target User ID")

    args = parser.parse_args()
    memory, agent, storage_name = get_system(provider=args.storage)

    if args.command == "chat" or args.command is None:
        user = getattr(args, "user", "alex_engineer")
        interactive_chat(memory, agent, default_user=user, storage_name=storage_name)
    elif args.command == "memories":
        cmd_memories(memory, args.user)
    elif args.command == "graph":
        cmd_graph(memory, args.user)
    elif args.command == "persona":
        cmd_persona(memory, args.user)
    elif args.command == "search":
        cmd_search(memory, args.user, args.query)
    elif args.command == "add":
        rep = memory.add(args.statement, user_id=args.user)
        print(f"{GREEN}Fact processed:{RESET} {rep}")
    elif args.command == "reset":
        memory.reset(args.user)
        print(f"{GREEN}Wiped memory for '{args.user}'.{RESET}")


if __name__ == "__main__":
    main()
