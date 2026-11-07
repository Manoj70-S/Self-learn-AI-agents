"""
CogniMem Interactive Web Dashboard & Chatbot
Powered by Streamlit.
Features live chat, real-time memory inspection, knowledge graph exploration, and persona card viewer.
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from cognimem import CogniMemory, CognitiveAgent, MemoryStatus
from cognimem.storage.sqlite_store import SQLiteVectorStore
from cognimem.embeddings import get_embedding_provider

# Page configuration
st.set_page_config(
    page_title="CogniMem - Cognitive Agent Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #4F46E5;
        margin-bottom: 10px;
    }
    .badge-active {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-superseded {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cogni_system():
    storage = SQLiteVectorStore(db_path="cognimem.db")
    embeddings = get_embedding_provider("deterministic", dimensions=1536)
    memory = CogniMemory(storage=storage, embeddings=embeddings, db_path="cognimem.db")
    agent = CognitiveAgent(memory=memory)
    return memory, agent


memory, agent = get_cogni_system()

# Sidebar: User & System Controls
st.sidebar.title("🧠 CogniMem Control")
st.sidebar.caption("Self-Evolving Cognitive Memory System")

user_id = st.sidebar.text_input("Active User ID", value="alex_engineer")
session_id = st.sidebar.text_input("Session ID", value="session_main")

st.sidebar.divider()
st.sidebar.subheader("Quick Stats")
active_memories = memory.get_all(user_id, status=MemoryStatus.ACTIVE)
superseded_memories = memory.get_all(user_id, status=MemoryStatus.SUPERSEDED)
relations = memory.graph.get_relations(user_id)

col_s1, col_s2 = st.sidebar.columns(2)
col_s1.metric("Active Facts", len(active_memories))
col_s2.metric("Superseded", len(superseded_memories))
st.sidebar.metric("Graph Triples", len(relations))

if st.sidebar.button("🧹 Wipe User Memory", type="secondary"):
    memory.reset(user_id)
    st.sidebar.success(f"Wiped all memories for {user_id}!")
    st.rerun()

# Main Application Tabs
tab_chat, tab_inspector, tab_graph, tab_persona = st.tabs([
    "💬 Interactive Agent Chat",
    "🔍 Memory Inspector (Audit Trail)",
    "🕸️ Knowledge Graph",
    "📇 Persona Card"
])

# =============================================================================
# TAB 1: Chat Interface
# =============================================================================
with tab_chat:
    st.subheader(f"Chat with Cognitive Agent (User: `{user_id}`)")
    st.caption("Agent automatically retrieves memories before responding, and consolidates new facts in the background.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display past messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("recalled"):
                with st.expander("⚡ Context Recalled by ACT-R Scoring"):
                    for r in msg["recalled"]:
                        st.markdown(f"- **Score {r.get('score', 0):.3f}**: {r.get('memory')}")

    # Chat Input
    if prompt := st.chat_input("Say something (e.g. 'I prefer FastAPI', 'What stack do you recommend for me?')"):
        # Append User Message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Call Agent
        with st.chat_message("assistant"):
            with st.spinner("Retrieving memory & reasoning..."):
                response = agent.chat(
                    user_message=prompt,
                    user_id=user_id,
                    session_id=session_id
                )
                reply = response["reply"]
                recalled = response.get("retrieved_memories", [])
                st.markdown(reply)

                if recalled:
                    with st.expander("⚡ Context Recalled by ACT-R Scoring", expanded=False):
                        for r in recalled:
                            st.markdown(f"- **Score {r.get('score', 0):.3f}**: {r.get('memory')}")

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": reply,
            "recalled": recalled
        })

# =============================================================================
# TAB 2: Memory Inspector
# =============================================================================
with tab_inspector:
    st.subheader("Persistent Vector Store Audit Trail")
    
    st.markdown("#### 🟢 Active Memories")
    if active_memories:
        active_data = []
        for m in active_memories:
            active_data.append({
                "ID": m.id[:8] + "...",
                "Content": m.content,
                "Type": m.memory_type if isinstance(m.memory_type, str) else m.memory_type.value,
                "Importance": round(getattr(m, "importance", 0.5), 2),
                "Access Count": m.access_count,
                "Created At": m.created_at.strftime("%Y-%m-%d %H:%M:%S")
            })
        st.dataframe(pd.DataFrame(active_data), use_container_width=True)
    else:
        st.info("No active memories found for this user yet. Start chatting to add facts!")

    st.markdown("#### 🔴 Superseded Memories (Conflict Provenance Trail)")
    if superseded_memories:
        sup_data = []
        for s in superseded_memories:
            sup_data.append({
                "ID": s.id[:8] + "...",
                "Content": s.content,
                "Superseded By (New ID)": (s.superseded_by[:8] + "...") if s.superseded_by else "N/A",
                "Last Updated": s.updated_at.strftime("%Y-%m-%d %H:%M:%S")
            })
        st.dataframe(pd.DataFrame(sup_data), use_container_width=True)
    else:
        st.caption("No memories have been superseded yet. (Update a fact to see conflict resolution in action!)")

# =============================================================================
# TAB 3: Knowledge Graph
# =============================================================================
with tab_graph:
    st.subheader("Relational Knowledge Graph Triples")
    st.caption("Subject-Predicate-Object triples linking entities for multi-hop reasoning.")

    if relations:
        graph_data = [
            {"Subject (Entity A)": r.subject, "Predicate (Relation)": f"--[{r.predicate}]-->", "Object (Entity B)": r.object, "Confidence": r.confidence}
            for r in relations
        ]
        st.dataframe(pd.DataFrame(graph_data), use_container_width=True)
        
        # Visual breakdown
        st.markdown("#### Entity Connectivity Network")
        ui_elements = memory.export_knowledge_graph(user_id)
        cols = st.columns(3)
        for i, edge in enumerate(ui_elements["edges"]):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <b>{edge['source']}</b><br>
                    <small><i>{edge['label']}</i></small><br>
                    ➔ <b>{edge['target']}</b>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Knowledge Graph is empty for this user. Introduce facts (e.g. 'I live in Berlin and work as a Python engineer') to generate triples.")

# =============================================================================
# TAB 4: Persona Card
# =============================================================================
with tab_persona:
    st.subheader("Synthesized Executive Persona Card")
    st.caption("Bounded user profile injected into LLM prompts for immediate grounding.")
    card_markdown = memory.get_persona_card(user_id)
    st.markdown(card_markdown)
