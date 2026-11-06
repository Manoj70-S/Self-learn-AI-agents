"""
CogniMem running with Qdrant Cloud Cluster as the Vector Storage Backend.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from cognimem import CogniMemory, CognitiveAgent
from cognimem.storage.qdrant_store import QdrantVectorStore
from cognimem.embeddings import get_embedding_provider
from cognimem.models import MemoryStatus


def main():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    print("=" * 70)
    print(" INITIALIZING COGNIMEM WITH QDRANT CLUSTER BACKEND")
    print("=" * 70)
    print(f"Qdrant Endpoint: {qdrant_url}")

    # Initialize QdrantVectorStore connected to user's Qdrant Cloud
    qdrant_store = QdrantVectorStore(
        collection_name="agent_memories",
        url=qdrant_url,
        api_key=qdrant_api_key,
        vector_size=1536
    )

    embeddings = get_embedding_provider("deterministic", dimensions=1536)

    memory = CogniMemory(storage=qdrant_store, embeddings=embeddings)
    agent = CognitiveAgent(memory=memory)

    USER_ID = "alex_cloud_qdrant"

    # Reset any previous test records for clean demo
    qdrant_store.clear(user_id=USER_ID)

    # Session 1: Onboarding
    print("\n--- Session 1: Onboarding ---")
    msg1 = "Hello! I am Alex. I live in Amsterdam and prefer FastAPI and PostgreSQL."
    print(f"User: {msg1}")
    res1 = agent.chat(user_message=msg1, user_id=USER_ID)
    print(f"Agent: {res1['reply']}")

    # Session 2: Days Later (Empty Context Query)
    print("\n--- Session 2: Days Later (Querying from Qdrant Cloud) ---")
    msg2 = "What database and framework should I use for my service?"
    print(f"User: {msg2}")
    res2 = agent.chat(user_message=msg2, user_id=USER_ID)
    print("Recalled from Qdrant Cloud:")
    for m in res2["retrieved_memories"]:
        print(f"  * {m['memory']} (Score: {m['score']:.4f})")
    print(f"Agent: {res2['reply']}")

    # Session 3: Conflict Evolution on Cloud
    print("\n--- Session 3: Conflict Evolution (Relocation) ---")
    msg3 = "I just moved to Berlin."
    print(f"User: {msg3}")
    res3 = agent.chat(user_message=msg3, user_id=USER_ID)
    print(f"Agent: {res3['reply']}")

    print("\nActive Memories in Qdrant Cloud:")
    active = memory.get_all(USER_ID, status=MemoryStatus.ACTIVE)
    for a in active:
        print(f"  [ACTIVE] {a.content}")

    print("\nSuperseded Memories in Qdrant Cloud (Audit Trail):")
    superseded = memory.get_all(USER_ID, status=MemoryStatus.SUPERSEDED)
    for s in superseded:
        print(f"  [SUPERSEDED] {s.content} -> Replaced by: {s.superseded_by}")

    print("\n" + "=" * 70)
    print(" QDRANT CLOUD END-TO-END PIPELINE SUCCESSFUL!")
    print("=" * 70)


if __name__ == "__main__":
    main()
