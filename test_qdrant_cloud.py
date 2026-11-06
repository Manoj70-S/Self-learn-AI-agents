"""
Test connectivity and operations with Qdrant Cloud cluster.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


def main():
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    print("=" * 65)
    print(" CONNECTING TO QDRANT CLOUD CLUSTER")
    print("=" * 65)
    print(f"Endpoint: {qdrant_url}")
    print(f"API Key:  {qdrant_api_key[:10]}...{qdrant_api_key[-10:]}")

    # Initialize Qdrant Client
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # 1. Fetch cluster collections
    print("\n--- 1. Fetching Collections ---")
    collections_response = client.get_collections()
    collections = [c.name for c in collections_response.collections]
    print(f"Existing Collections on Cluster: {collections}")

    COLLECTION_NAME = "agent_memories"
    VECTOR_SIZE = 1536

    # 2. Ensure agent_memories collection and payload indexes exist
    if COLLECTION_NAME not in collections:
        print(f"\n--- 2. Creating collection '{COLLECTION_NAME}' (dims={VECTOR_SIZE}) ---")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qmodels.VectorParams(size=VECTOR_SIZE, distance=qmodels.Distance.COSINE)
        )
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
    else:
        print(f"\n--- 2. Collection '{COLLECTION_NAME}' already exists on cluster. ---")

    print("Creating keyword payload indexes for 'user_id' and 'status'...")
    for field in ("user_id", "status"):
        try:
            client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD
            )
        except Exception as e:
            pass

    # 3. Test Upserting a sample memory vector
    print("\n--- 3. Testing Cloud Vector Upsert ---")
    sample_point_id = "00000000-0000-0000-0000-000000000001"
    sample_vector = [0.01] * VECTOR_SIZE
    sample_payload = {
        "user_id": "alex_qdrant_user",
        "content": "User prefers FastAPI and PostgreSQL on AWS.",
        "category": "tech_stack",
        "status": "active"
    }

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            qmodels.PointStruct(
                id=sample_point_id,
                vector=sample_vector,
                payload=sample_payload
            )
        ]
    )
    print("Upsert succeeded: Point written to Qdrant Cloud!")

    # 4. Test Querying / Searching
    print("\n--- 4. Testing Semantic Search on Qdrant Cloud ---")
    query_vector = [0.01] * VECTOR_SIZE
    search_filter = qmodels.Filter(
        must=[qmodels.FieldCondition(key="user_id", match=qmodels.MatchValue(value="alex_qdrant_user"))]
    )

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=search_filter,
            limit=3
        )
        search_res = response.points
    else:
        search_res = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=3
        )

    print(f"Search Results (Found {len(search_res)} hits):")
    for hit in search_res:
        print(f"[*] ID: {hit.id} | Score: {hit.score:.4f} | Payload: {hit.payload}")

    print("\n" + "=" * 65)
    print(" QDRANT CLOUD TEST COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
