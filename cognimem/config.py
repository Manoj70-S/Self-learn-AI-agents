import os
from dotenv import load_dotenv

load_dotenv()

# Default configuration dictionary
DEFAULT_CONFIG = {
    "storage": {
        "provider": os.getenv("COGNIMEM_STORAGE_PROVIDER", "sqlite"),  # "sqlite" or "qdrant"
        "sqlite_path": os.getenv("COGNIMEM_DB_PATH", "cognimem.db"),
        "qdrant": {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", 6333)),
            "collection_name": os.getenv("QDRANT_COLLECTION", "agent_memories"),
            "url": os.getenv("QDRANT_URL", None),
            "api_key": os.getenv("QDRANT_API_KEY", None),
        }
    },
    "llm": {
        "provider": "openai",
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "api_key": os.getenv("OPENAI_API_KEY", None)
    },
    "embeddings": {
        "provider": "openai" if os.getenv("OPENAI_API_KEY") else "deterministic",
        "model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        "dimensions": 1536
    }
}
