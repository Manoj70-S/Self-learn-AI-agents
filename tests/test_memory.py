import pytest
import os
from cognimem import CogniMemory
from cognimem.models import MemoryStatus, MemoryItem, MemoryType
from cognimem.storage.sqlite_store import SQLiteVectorStore
from cognimem.embeddings import DeterministicEmbeddings


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_memory.db")
    return db_file


@pytest.fixture
def memory_instance(temp_db):
    embeddings = DeterministicEmbeddings(dimensions=128)
    storage = SQLiteVectorStore(db_path=temp_db)
    return CogniMemory(storage=storage, embeddings=embeddings)


def test_sqlite_vector_crud(temp_db):
    storage = SQLiteVectorStore(db_path=temp_db)
    item = MemoryItem(
        id="item-1",
        user_id="user_test",
        content="User loves Python",
        memory_type=MemoryType.PREFERENCE,
        embedding=[0.1] * 128
    )
    storage.upsert(item)

    fetched = storage.get("item-1")
    assert fetched is not None
    assert fetched.content == "User loves Python"
    assert fetched.user_id == "user_test"

    # Search
    results = storage.search(query_vector=[0.1] * 128, user_id="user_test", limit=5)
    assert len(results) == 1
    assert results[0][0].id == "item-1"

    # Update status
    storage.update_status("item-1", MemoryStatus.SUPERSEDED, superseded_by="item-2")
    active_items = storage.get_all("user_test", status=MemoryStatus.ACTIVE)
    assert len(active_items) == 0

    superseded_items = storage.get_all("user_test", status=MemoryStatus.SUPERSEDED)
    assert len(superseded_items) == 1
    assert superseded_items[0].superseded_by == "item-2"


def test_fact_extraction_and_addition(memory_instance):
    input_text = "I prefer FastAPI and PostgreSQL. I am allergic to peanuts."
    report = memory_instance.add(input_text, user_id="user_1")

    assert report["extracted_count"] >= 2
    assert len(report["added"]) >= 2

    # Check search
    matches = memory_instance.search("What is the user's allergy?", user_id="user_1")
    assert len(matches) > 0
    assert any("peanut" in m["memory"].lower() for m in matches)


def test_deduplication(memory_instance):
    # Add initial fact
    memory_instance.add("I prefer FastAPI", user_id="user_1")
    initial_count = len(memory_instance.get_all(user_id="user_1"))

    # Add the exact same statement again
    report2 = memory_instance.add("I prefer FastAPI", user_id="user_1")
    final_count = len(memory_instance.get_all(user_id="user_1"))

    # Should not duplicate
    assert final_count == initial_count
    assert len(report2["ignored"]) > 0


def test_conflict_resolution(memory_instance):
    # 1. User lives in Amsterdam
    memory_instance.add("I live in Amsterdam", user_id="user_1")
    active = memory_instance.get_all(user_id="user_1", status=MemoryStatus.ACTIVE)
    assert any("Amsterdam" in m.content for m in active)

    # 2. User moves to Berlin
    report = memory_instance.add("I moved to Berlin", user_id="user_1")

    # The old memory should now be superseded
    active_after = memory_instance.get_all(user_id="user_1", status=MemoryStatus.ACTIVE)
    assert any("Berlin" in m.content for m in active_after)
    assert not any("Amsterdam" in m.content for m in active_after)

    superseded = memory_instance.get_all(user_id="user_1", status=MemoryStatus.SUPERSEDED)
    assert len(superseded) >= 1
    assert any("Amsterdam" in s.content for s in superseded)


def test_multi_user_isolation(memory_instance):
    memory_instance.add("I prefer Rust", user_id="user_alice")
    memory_instance.add("I prefer Python", user_id="user_bob")

    alice_mems = memory_instance.search("programming language", user_id="user_alice")
    bob_mems = memory_instance.search("programming language", user_id="user_bob")

    assert all("Rust" in m["memory"] for m in alice_mems)
    assert not any("Python" in m["memory"] for m in alice_mems)

    assert all("Python" in m["memory"] for m in bob_mems)
    assert not any("Rust" in m["memory"] for m in bob_mems)


def test_knowledge_graph_relational_triples(memory_instance):
    # Add input with entity relationships
    memory_instance.add("My name is Alex. I live in Berlin and work with FastAPI.", user_id="user_graph")
    
    relations = memory_instance.graph.get_relations("user_graph")
    assert len(relations) >= 2
    
    predicates = [r.predicate for r in relations]
    assert any("LIVES_IN" in p for p in predicates) or any("NAME_IS" in p for p in predicates)
    
    # Check multi-hop connected entities
    connected = memory_instance.graph.get_connected_entities("User", user_id="user_graph")
    assert len(connected) >= 1


def test_persona_card_generation(memory_instance):
    memory_instance.add(
        "I am strictly allergic to peanuts. I prefer FastAPI and PostgreSQL.",
        user_id="user_persona"
    )
    card = memory_instance.get_persona_card("user_persona")
    assert "COGNITIVE PERSONA" in card
    assert "FastAPI" in card or "Peanuts" in card


def test_act_r_cognitive_scoring(memory_instance):
    # Allergy has higher importance (0.95) than casual comment
    memory_instance.add("I am allergic to shellfish.", user_id="user_scoring")
    results = memory_instance.search("shellfish", user_id="user_scoring")
    assert len(results) > 0
    assert results[0]["importance"] >= 0.8
    assert results[0]["score"] > 0.40
