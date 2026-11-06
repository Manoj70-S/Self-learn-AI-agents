from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import uuid
try:
    from pydantic import BaseModel, Field, ConfigDict
except ImportError:
    from pydantic import BaseModel, Field
    ConfigDict = None


class MemoryType(str, Enum):
    """Categorization of memory types according to cognitive science."""
    PROFILE = "profile"        # Persistent user identity, constraints, immutable traits
    PREFERENCE = "preference"  # Likes, dislikes, tools, coding styles, language choices
    EPISODIC = "episodic"      # Specific past events, milestones, or conversations
    SEMANTIC = "semantic"      # General facts, learned domain knowledge, project rules


class MemoryStatus(str, Enum):
    """Lifecycle status of a memory record."""
    ACTIVE = "active"          # Currently valid and searchable
    SUPERSEDED = "superseded"  # Replaced by newer contradictory knowledge
    ARCHIVED = "archived"      # Outdated or suppressed


class ReconciliationAction(str, Enum):
    """Action determined during the memory consolidation phase."""
    ADD = "add"                # Distinct, novel fact -> insert as new memory
    UPDATE = "update"          # Contradicts/updates old fact -> replace old, insert new
    DELETE = "delete"          # Fact was explicitly revoked -> mark inactive/remove
    NOOP = "noop"              # Fact is identical or redundant -> do nothing


class ExtractedFact(BaseModel):
    """Atomic fact extracted from conversation turns."""
    fact: str = Field(description="Normalized, self-contained atomic statement in third person")
    memory_type: MemoryType = Field(default=MemoryType.PREFERENCE, description="Type of memory")
    category: str = Field(default="general", description="Subcategory: tech_stack, dietary, habit, etc.")
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0")
    temporal_context: Optional[str] = Field(default=None, description="Time context, e.g. 'recently', 'always'")


class ReconciliationDecision(BaseModel):
    """Structured decision made by the cognitive reconciler."""
    action: ReconciliationAction
    target_memory_id: Optional[str] = None
    resolved_fact: Optional[str] = None
    rationale: str = Field(default="", description="Reasoning for this reconciliation decision")


class MemoryItem(BaseModel):
    """Core memory unit stored in the persistent vector store."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    content: str
    memory_type: MemoryType = MemoryType.PREFERENCE
    status: MemoryStatus = MemoryStatus.ACTIVE
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    superseded_by: Optional[str] = None
    agent_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    if ConfigDict is not None:
        model_config = ConfigDict(use_enum_values=True)
    else:
        class Config:
            use_enum_values = True
