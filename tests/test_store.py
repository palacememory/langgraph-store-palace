"""Integration tests against a real `palace` binary in a throwaway data dir.

Run: PALACE_BIN=/path/to/palace pytest -q
"""

import asyncio
import shutil
import os

import pytest

from langgraph_store_palace import PalaceStore

PALACE = os.environ.get("PALACE_BIN") or shutil.which("palace")
pytestmark = pytest.mark.skipif(not PALACE, reason="palace binary not installed")


@pytest.fixture()
def store(tmp_path):
    s = PalaceStore(palace_bin=PALACE, palace_path=str(tmp_path / "palace"), wing="lg_test")
    yield s
    s.close()


def test_put_get_overwrite_delete(store):
    ns = ("users", "alice", "memories")
    store.put(ns, "editor", {"text": "Alice uses Neovim with tabs, never spaces.", "kind": "pref"})
    item = store.get(ns, "editor")
    assert item is not None and item.value["kind"] == "pref"
    created = item.created_at

    store.put(ns, "editor", {"text": "Alice switched to Helix in November.", "kind": "pref"})
    item = store.get(ns, "editor")
    assert "Helix" in item.value["text"]
    assert item.created_at == created and item.updated_at >= created

    store.delete(ns, "editor")
    assert store.get(ns, "editor") is None


def test_semantic_search_prefix_and_filter(store):
    store.put(("users", "alice", "memories"), "queue",
              {"text": "We moved the job queue from Redis to SQLite because the deploy target has no persistent volume.", "kind": "decision"})
    store.put(("users", "alice", "memories"), "lunch",
              {"text": "Alice likes ramen for lunch on Fridays.", "kind": "pref"})
    store.put(("users", "bob", "memories"), "queue",
              {"text": "Bob wants to keep Redis for the cache layer.", "kind": "decision"})

    hits = store.search(("users", "alice"), query="why did we drop Redis for the queue?", limit=2)
    assert hits and hits[0].key == "queue"
    assert all(h.namespace[:2] == ("users", "alice") for h in hits)
    assert hits[0].score is not None

    prefs = store.search(("users",), filter={"kind": "pref"})
    assert [h.key for h in prefs] == ["lunch"]


def test_list_namespaces(store):
    store.put(("users", "alice", "memories"), "a", {"text": "one fact about alice"})
    store.put(("users", "bob", "notes"), "b", {"text": "a note from bob about deploys"})
    assert set(store.list_namespaces(prefix=("users",))) == {
        ("users", "alice", "memories"), ("users", "bob", "notes")}
    assert set(store.list_namespaces(max_depth=2)) == {("users", "alice"), ("users", "bob")}
    assert store.list_namespaces(suffix=("notes",)) == [("users", "bob", "notes")]


def test_index_false_and_async(store):
    async def run():
        await store.aput(("cfg",), "raw", {"blob": [1, 2, 3]}, index=False)
        item = await store.aget(("cfg",), "raw")
        assert item.value == {"blob": [1, 2, 3]}
        assert await store.asearch(("cfg",), query="blob") == []
    asyncio.run(run())


def test_in_a_graph(store):
    from langgraph.graph import StateGraph, START, END
    from langgraph.runtime import Runtime  # noqa: F401
    from typing import TypedDict

    class State(TypedDict):
        question: str
        answer: str

    store.put(("project", "api"), "auth", {"text": "Auth tokens expire after 15 minutes; refresh via /token/refresh."})

    def recall(state: State, *, store) -> dict:  # LangGraph injects the compiled store
        hits = store.search(("project",), query=state["question"], limit=1)
        return {"answer": hits[0].value["text"] if hits else ""}

    g = StateGraph(State)
    g.add_node("recall", recall)
    g.add_edge(START, "recall")
    g.add_edge("recall", END)
    app = g.compile(store=store)
    out = app.invoke({"question": "how long do auth tokens last?", "answer": ""})
    assert "15 minutes" in out["answer"]
