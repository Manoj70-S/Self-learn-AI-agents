"""
CogniMem Demonstration Script:
Showcases multi-session persistent memory, semantic retrieval, and conflict resolution.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from cognimem import CogniMemory, CognitiveAgent
from cognimem.models import MemoryStatus


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print_separator("INITIALIZING COGNIMEM SYSTEM")
    
    # Initialize CogniMemory with local SQLite persistence (cognimem_demo.db)
    db_file = "demo_memory.db"
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except Exception:
            pass

    memory = CogniMemory(db_path=db_file)
    agent = CognitiveAgent(memory=memory)

    USER_ID = "alex_engineer"

    # =========================================================================
    # SESSION 1: First Onboarding
    # =========================================================================
    print_separator("SESSION 1: First Conversation (Onboarding)")
    session_1_msg = (
        "Hi! My name is Alex. I live in Amsterdam and work as a Python backend developer. "
        "I prefer FastAPI and PostgreSQL for building REST APIs. "
        "Also, I am strictly allergic to peanuts."
    )
    print(f"[User]: {session_1_msg}\n")

    res1 = agent.chat(user_message=session_1_msg, user_id=USER_ID, session_id="sess_001")
    print(f"[Agent]: {res1['reply']}\n")

    print(">> Memory Consolidation Audit Report:")
    for added in res1["consolidation"]["added"]:
        print(f"   [+] Fact Stored: '{added['fact']}' (ID: {added['id'][:8]}...)")

    # =========================================================================
    # SESSION 2: Independent Session Days Later (Empty Short-Term Context)
    # =========================================================================
    print_separator("SESSION 2: Days Later (Zero Short-Term History)")
    print("Simulating a fresh session where short-term dialogue context is completely empty.\n")

    # Query 1: Tech recommendation
    query_tech = "What framework and database would you recommend for my new service?"
    print(f"[User]: {query_tech}\n")
    res_tech = agent.chat(user_message=query_tech, user_id=USER_ID, session_id="sess_002")
    
    print(">> Memories Recalled by Semantic Vector Search:")
    for mem in res_tech["retrieved_memories"]:
        print(f"   [*] Score: {mem['score']} | {mem['memory']}")
    print(f"\n[Agent]: {res_tech['reply']}\n")

    # Query 2: Dietary recommendation
    query_food = "Can you recommend a quick Thai dish for dinner tonight?"
    print(f"[User]: {query_food}\n")
    res_food = agent.chat(user_message=query_food, user_id=USER_ID, session_id="sess_002")

    print(">> Memories Recalled by Semantic Vector Search:")
    for mem in res_food["retrieved_memories"]:
        print(f"   [*] Score: {mem['score']} | {mem['memory']}")
    print(f"\n[Agent]: {res_food['reply']}\n")

    # =========================================================================
    # SESSION 3: Conflict Resolution & Memory Evolution
    # =========================================================================
    print_separator("SESSION 3: Preference Conflict & Updating (Weeks Later)")
    session_3_msg = "Just a quick update: I moved to Berlin and switched from FastAPI to Go for my backend."
    print(f"[User]: {session_3_msg}\n")

    res3 = agent.chat(user_message=session_3_msg, user_id=USER_ID, session_id="sess_003")
    print(f"[Agent]: {res3['reply']}\n")

    print(">> Consolidation Actions Taken (Contradiction Resolution):")
    for updated in res3["consolidation"]["updated"]:
        print(f"   [^] Updated Fact: '{updated['fact']}'")
        print(f"       Superseded Old Memory ID: {updated.get('superseded_id')}")

    for added in res3["consolidation"]["added"]:
        print(f"   [+] Novel Fact Added: '{added['fact']}'")

    # =========================================================================
    # AUDIT: Active Profile vs Superseded Knowledge
    # =========================================================================
    print_separator("AUDIT: Final Cognitive State for User 'alex_engineer'")
    
    profile = memory.get_profile(USER_ID)
    print("ACTIVE USER PROFILE:")
    for section, items in profile.items():
        if items:
            print(f"  * {section.upper()}:")
            for item in items:
                print(f"     - {item}")

    print("\nHISTORICAL / SUPERSEDED MEMORIES (Preserved for Audit Trail):")
    superseded_items = memory.get_all(USER_ID, status=MemoryStatus.SUPERSEDED)
    if superseded_items:
        for s in superseded_items:
            print(f"  [SUPERSEDED] '{s.content}' -> Replaced by: {s.superseded_by}")
    else:
        print("  None.")

    print_separator("DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    main()
