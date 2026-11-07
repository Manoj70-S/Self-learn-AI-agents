"""
FastAPI REST API Service for CogniMem.
Exposes cognitive memory search, chat, graph queries, and persona endpoints.
Run with: uvicorn server:app --host 0.0.0.0 --port 8000
"""

from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cognimem import CogniMemory, CognitiveAgent, MemoryStatus
from cognimem.storage.sqlite_store import SQLiteVectorStore
from cognimem.embeddings import get_embedding_provider

app = FastAPI(
    title="CogniMem REST API",
    description="Cognitive Long-Term Memory Engine & Knowledge Graph API for AI Agents",
    version="2.0.0"
)

# Global memory & agent instances
storage = SQLiteVectorStore(db_path="cognimem.db")
embeddings = get_embedding_provider("deterministic", dimensions=1536)
memory = CogniMemory(storage=storage, embeddings=embeddings, db_path="cognimem.db")
agent = CognitiveAgent(memory=memory)


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique ID of the user")
    message: str = Field(..., description="User input message")
    session_id: Optional[str] = Field(default="default_session")
    async_consolidation: bool = Field(default=False, description="Queue memory extraction asynchronously")


class AddMemoryRequest(BaseModel):
    content: str = Field(..., description="Statement to index into memory")
    category: Optional[str] = Field(default="general")
    importance: Optional[float] = Field(default=0.5)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "CogniMem", "version": "2.0.0"}


@app.post("/api/chat")
def chat_turn(req: ChatRequest):
    """Execute a context-aware chat turn with automatic memory consolidation."""
    result = agent.chat(
        user_message=req.message,
        user_id=req.user_id,
        session_id=req.session_id,
        async_consolidation=req.async_consolidation
    )
    return result


@app.get("/api/memories/{user_id}")
def get_user_memories(user_id: str, status: Optional[str] = "active"):
    """Fetch memories for a user filtered by status (active or superseded)."""
    mem_status = MemoryStatus(status.lower()) if status else None
    items = memory.get_all(user_id, status=mem_status)
    return {
        "user_id": user_id,
        "count": len(items),
        "memories": [
            {
                "id": m.id,
                "content": m.content,
                "type": m.memory_type if isinstance(m.memory_type, str) else m.memory_type.value,
                "importance": getattr(m, "importance", 0.5),
                "access_count": m.access_count,
                "status": m.status if isinstance(m.status, str) else m.status.value,
                "superseded_by": m.superseded_by,
                "created_at": m.created_at.isoformat()
            }
            for m in items
        ]
    }


@app.post("/api/memories/{user_id}")
def add_memory(user_id: str, req: AddMemoryRequest):
    """Directly insert a declarative statement into a user's memory."""
    report = memory.add(req.content, user_id=user_id)
    return {"status": "success", "report": report}


@app.delete("/api/memories/{memory_id}")
def delete_memory(memory_id: str):
    """Permanently delete a specific memory by ID."""
    success = memory.delete(memory_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory ID not found")
    return {"status": "deleted", "id": memory_id}


@app.get("/api/persona/{user_id}")
def get_user_persona(user_id: str):
    """Synthesize an Executive Persona Card for the user."""
    card = memory.get_persona_card(user_id)
    profile = memory.get_profile(user_id)
    return {
        "user_id": user_id,
        "persona_markdown": card,
        "structured_profile": profile
    }


@app.get("/api/graph/{user_id}")
def get_user_graph(user_id: str):
    """Retrieve Knowledge Graph nodes and edges for the user."""
    relations = memory.graph.get_relations(user_id)
    elements = memory.export_knowledge_graph(user_id)
    return {
        "user_id": user_id,
        "triple_count": len(relations),
        "graph_elements": elements,
        "triples": [
            {"subject": r.subject, "predicate": r.predicate, "object": r.object, "confidence": r.confidence}
            for r in relations
        ]
    }


@app.delete("/api/reset/{user_id}")
def reset_user(user_id: str):
    """Wipe all memories and graph triples for a user."""
    count = memory.reset(user_id)
    return {"status": "reset", "user_id": user_id, "cleared_records": count}
