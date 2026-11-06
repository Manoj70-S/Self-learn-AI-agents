"""
Mem0 Cloud Platform Integration Demo
Uses your Mem0 API Key to interact directly with the managed Mem0 cloud service.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from mem0 import MemoryClient


def main():
    api_key = os.getenv("MEM0_API_KEY")
    if not api_key:
        print("ERROR: MEM0_API_KEY is not set in your .env file.")
        return

    print("=" * 60)
    print(" CONNECTING TO MEM0 CLOUD PLATFORM")
    print("=" * 60)
    print(f"Using API Key: {api_key[:6]}...{api_key[-4:]}")

    # Initialize client
    client = MemoryClient(api_key=api_key)
    USER_ID = "alex_cloud_user"

    import time

    # Step 1: Add initial conversation
    print("\n--- 1. Adding Conversation to Mem0 Cloud ---")
    messages = [
        {"role": "user", "content": "Hi, I am Alex. I live in Amsterdam and work as a Python engineer."},
        {"role": "assistant", "content": "Nice to meet you, Alex! How can I assist you today?"},
        {"role": "user", "content": "I prefer using FastAPI and PostgreSQL. I also have a severe allergy to peanuts."}
    ]
    add_response = client.add(messages, user_id=USER_ID)
    print("Mem0 Cloud Add Response:")
    print(add_response)

    print("\nWaiting 4 seconds for Mem0 Cloud background fact extraction to complete...")
    time.sleep(4)

    # Step 2: Search relevant memories
    print("\n--- 2. Semantic Search on Mem0 Cloud ---")
    query = "What food allergy does the user have?"
    search_results = client.search(query, filters={"user_id": USER_ID})
    print(f"Query: '{query}'")
    for mem in search_results.get("results", []):
        print(f"[*] Memory: {mem.get('memory')} (ID: {mem.get('id')})")

    # Step 3: List all memories stored for this user
    print("\n--- 3. Listing All Active Memories on Mem0 Cloud ---")
    all_memories = client.get_all(filters={"user_id": USER_ID})
    for mem in all_memories.get("results", []):
        print(f"[-] {mem.get('memory')} | Created: {mem.get('created_at')}")

    print("\n" + "=" * 60)
    print(" MEM0 CLOUD DEMO COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
