# langgraph-store-palace

[![tests](https://github.com/palacememory/langgraph-store-palace/actions/workflows/test.yml/badge.svg)](https://github.com/palacememory/langgraph-store-palace/actions/workflows/test.yml) [![Star palace-rs](https://img.shields.io/github/stars/AncientiCe/palace-rs?style=social&label=Star%20palace-rs)](https://github.com/AncientiCe/palace-rs)

> **Palace core lives at [AncientiCe/palace-rs](https://github.com/AncientiCe/palace-rs).** This package is a thin adapter; all the retrieval work happens in the `palace` binary. If it is useful to you, please [star palace-rs on GitHub](https://github.com/AncientiCe/palace-rs). Stars are how other developers find it. Website: [palacememory.com](https://palacememory.com).

A LangGraph `BaseStore` backed by [Palace](https://github.com/AncientiCe/palace-rs): long-term agent memory that stays on your machine, stores text verbatim, and answers semantic queries with hybrid vector and BM25 search. One Rust binary, one SQLite file, no API keys, no cloud.

Not associated in any way with the popular mempalace python.

## Why

LangGraph's checkpointer keeps state inside one thread. The `store` is what survives across threads. Most store backends need a database server and an embeddings API. Palace gives you local embeddings (MiniLM on ONNX Runtime, on your CPU) and hybrid retrieval in a single binary, and it keeps the original text instead of an LLM summary.

## Install

1. Install the `palace` binary (Linux x86_64, macOS Apple Silicon, Windows):

   ```bash
   curl -fsSL https://raw.githubusercontent.com/AncientiCe/palace-rs/main/scripts/install.sh | sh
   # or: cargo install palace-rs
   palace doctor
   ```

   The install script also runs `palace install --all` to wire Palace into your coding agents. If you only want the binary for this store, that step is harmless, and `palace uninstall` undoes it.

2. Install the store from PyPI:

   ```bash
   pip install langgraph-store-palace
   ```

## Use

```python
from langgraph_store_palace import PalaceStore

store = PalaceStore()  # palace on PATH, data in ~/.palace/langgraph

ns = ("users", "alice", "memories")
store.put(ns, "queue", {"text": "We moved the job queue to SQLite because the deploy target has no persistent volume.", "kind": "decision"})
store.put(ns, "editor", {"text": "Alice uses Neovim with tabs.", "kind": "pref"})

store.get(ns, "editor").value
# {'text': 'Alice uses Neovim with tabs.', 'kind': 'pref'}

hits = store.search(("users", "alice"), query="why did we drop Redis for the queue?", limit=3)
hits[0].key, hits[0].score
# ('queue', 0.74)

store.search(("users",), filter={"kind": "pref"})       # exact filter, no query
store.list_namespaces(prefix=("users",), max_depth=2)   # [('users', 'alice')]
store.delete(ns, "editor")
```

In a graph:

```python
from langgraph.graph import StateGraph, START, END

def recall(state, *, store):
    hits = store.search(("project",), query=state["question"], limit=3)
    return {"context": [h.value["text"] for h in hits]}

graph = builder.compile(store=store)
```

Async methods (`aput`, `aget`, `asearch`, `alist_namespaces`, `adelete`) work too; they run the sync path in a worker thread.

## Options

| Argument | Default | Meaning |
|---|---|---|
| `wing` | `"langgraph"` | Palace wing that holds this store's drawers. |
| `palace_path` | `~/.palace/langgraph` | Data directory for the `palace.db` this store uses. A separate palace keeps app memory out of your coding agents' automatic recall. |
| `use_main_palace` | `False` | Share your main palace (`~/.palace/palace` or `$PALACE_PALACE_PATH`) instead. |
| `fields` | `None` | JSON paths to send to Palace for search, for example `["text", "notes[*]"]`. By default the first string in `text`, `content`, `memory` or `data` is used, else the whole value as JSON. `put(..., index=False)` skips search for one item. |
| `palace_bin` | `$PALACE_BIN` or `palace` on PATH | Path to the binary. |
| `oversample` | `5` | Palace hits fetched per requested result, so namespace and filter checks still leave enough results. |
| `index_db` | next to `palace.db` | Sidecar SQLite index for keys, JSON values and timestamps. |

## How it works

- `PalaceStore` starts one `palace mcp` process (stdio JSON-RPC) and keeps it alive. Calls are serialised with a lock, so one store can be shared across threads.
- Exact key/value semantics (namespaces, keys, JSON values, `created_at`, `updated_at`, filters, `list_namespaces`) live in a small sidecar SQLite table. That keeps `get` O(1) and filters exact.
- The text of each item is filed in Palace as a drawer (`palace_add_drawer`) in your wing, with the room derived from the namespace and `source_file` set to `langgraph://<namespace>#<key>` for provenance. Overwrites use `palace_update_drawer`; deletes use `palace_delete_drawer`.
- `search(prefix, query=...)` calls `palace_search` in the wing, maps hits back to keys under the prefix, applies the filter, and sets `SearchItem.score` to Palace's combined score.
- Palace refuses near-duplicate drawers (similarity 0.9 or more). Two keys with near-identical text share one drawer, which is only deleted when no key points at it any more.

## Limits

- TTL is not supported (`supports_ttl = False`).
- Scores are Palace's hybrid score, not raw cosine, so they are comparable within one query, not across stores.
- `search` without `query` scans the sidecar index for the prefix. That is fine for thousands of items per prefix; it is not a data warehouse.

## Tests

```bash
git clone https://github.com/palacememory/langgraph-store-palace && cd langgraph-store-palace
pip install -e '.[test]'
PALACE_BIN=$(which palace) pytest -q
```

CI downloads the latest `palace` release from [palace-rs releases](https://github.com/AncientiCe/palace-rs/releases) and runs the same tests on every push.

The tests run against a real `palace` binary in a throwaway data directory (put/get/overwrite/delete, semantic search with prefix and filter, `list_namespaces` prefix/suffix/depth, `index=False`, async, and a compiled `StateGraph` using the store).

## License

MIT, the same license as Palace core ([AncientiCe/palace-rs](https://github.com/AncientiCe/palace-rs)).

Bugs in this adapter: [open an issue here](https://github.com/palacememory/langgraph-store-palace/issues). Bugs in retrieval or the `palace` binary: [palace-rs issues](https://github.com/AncientiCe/palace-rs/issues).

If this saved you a database server, please [star palace-rs](https://github.com/AncientiCe/palace-rs).
