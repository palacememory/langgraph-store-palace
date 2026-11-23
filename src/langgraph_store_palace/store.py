"""LangGraph BaseStore backed by Palace (palace-rs).

Palace stores text verbatim in a local SQLite file and answers semantic
queries with hybrid vector + BM25 search, with no API keys and no cloud on the
default path. This store keeps LangGraph's exact key/value contract in a small
sidecar SQLite index and sends the text of each item to Palace as a drawer, so
`store.search(namespace, query=...)` is answered by Palace's retrieval.

Not associated in any way with the popular mempalace python.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    get_text_at_path,
    tokenize_path,
)

from ._mcp import PalaceError, PalaceMCPClient

_DEFAULT_TEXT_KEYS = ("text", "content", "memory", "data")
_SEP = "\x1f"  # unit separator, never valid inside a LangGraph namespace label


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ns_key(namespace: tuple[str, ...]) -> str:
    return _SEP.join(namespace)


def _ns_from_key(key: str) -> tuple[str, ...]:
    return tuple(key.split(_SEP)) if key else ()


def _room_for(namespace: tuple[str, ...]) -> str:
    room = "__".join(re.sub(r"[^A-Za-z0-9_.-]+", "-", part) for part in namespace)
    return (room or "general")[:120]


def _apply_op(item_value: Any, op: str, target: Any) -> bool:
    try:
        if op == "$eq":
            return item_value == target
        if op == "$ne":
            return item_value != target
        if op == "$gt":
            return item_value > target
        if op == "$gte":
            return item_value >= target
        if op == "$lt":
            return item_value < target
        if op == "$lte":
            return item_value <= target
        if op == "$in":
            return item_value in target
        if op == "$nin":
            return item_value not in target
    except TypeError:
        return False
    raise ValueError(f"Unsupported filter operator: {op}")


def _matches_filter(value: dict[str, Any], flt: dict[str, Any] | None) -> bool:
    if not flt:
        return True
    for key, expected in flt.items():
        actual = value.get(key)
        if isinstance(expected, dict) and any(k.startswith("$") for k in expected):
            if not all(_apply_op(actual, op, tgt) for op, tgt in expected.items()):
                return False
        elif isinstance(expected, dict):
            if not isinstance(actual, dict) or not _matches_filter(actual, expected):
                return False
        elif actual != expected:
            return False
    return True


def _ns_matches(ns: tuple[str, ...], cond: MatchCondition) -> bool:
    path = tuple(cond.path)
    if len(ns) < len(path):
        return False
    pairs = zip(ns, path) if cond.match_type == "prefix" else zip(reversed(ns), reversed(path))
    return all(p == "*" or n == p for n, p in pairs)


class PalaceStore(BaseStore):
    """A LangGraph long-term memory store that uses Palace for retrieval.

    Args:
        wing: Palace wing that holds this store's drawers. Default "langgraph",
            which keeps app memory apart from your coding-agent memory.
        index_db: Path of the sidecar index (keys, JSON values, timestamps).
            Default: ``langgraph_store.db`` next to the palace.db in use.
        fields: JSON paths whose text is sent to Palace for search. Default:
            the first string found in ``text``, ``content``, ``memory`` or
            ``data``, otherwise the whole value as JSON.
        palace_bin: Path to the ``palace`` binary (default: ``$PALACE_BIN`` or PATH).
        palace_path: Data directory for the palace.db this store uses. Default
            ``~/.palace/langgraph``, a palace of its own, so app memory does not
            show up in your coding agents' automatic recall. Set
            ``use_main_palace=True`` to share your main palace instead.
        use_main_palace: Use the main palace (``$PALACE_PALACE_PATH`` or
            ``~/.palace/palace``) instead of a separate one.
        oversample: How many Palace hits to fetch per requested result, so
            namespace and filter checks still leave enough results.
    """

    supports_ttl = False

    def __init__(
        self,
        *,
        wing: str = "langgraph",
        index_db: str | os.PathLike[str] | None = None,
        fields: list[str] | None = None,
        palace_bin: str | None = None,
        palace_path: str | None = None,
        use_main_palace: bool = False,
        oversample: int = 5,
        client: PalaceMCPClient | None = None,
    ) -> None:
        self.wing = wing
        self.fields = fields
        self._field_paths = [tokenize_path(f) for f in fields] if fields else None
        self.oversample = max(1, oversample)
        if palace_path is None and not use_main_palace:
            palace_path = str(Path.home() / ".palace" / "langgraph")
        self._client = client or PalaceMCPClient(palace_bin, palace_path=palace_path)
        if index_db is None:
            base = palace_path or os.environ.get("PALACE_PALACE_PATH") or str(
                Path.home() / ".palace" / "palace"
            )
            index_db = Path(base) / "langgraph_store.db"
        Path(index_db).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(index_db), check_same_thread=False)
        self._db_lock = threading.Lock()
        with self._db_lock, self._db:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS items (
                    ns TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    drawer_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (ns, key)
                )"""
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS items_drawer ON items(drawer_id)")

    # -- context management ------------------------------------------------
    def close(self) -> None:
        self._client.close()
        with self._db_lock:
            self._db.close()

    def __enter__(self) -> "PalaceStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- BaseStore -----------------------------------------------------------
    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, PutOp):
                self._put(op)
                results.append(None)
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append(self._list_namespaces(op))
            else:
                raise ValueError(f"Unknown operation: {op!r}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))

    # -- helpers -------------------------------------------------------------
    def _row_to_item(self, row: tuple, score: float | None = None, search: bool = False):
        ns, key, value, _drawer, created, updated = row
        kwargs = dict(
            namespace=_ns_from_key(ns),
            key=key,
            value=json.loads(value),
            created_at=datetime.fromisoformat(created),
            updated_at=datetime.fromisoformat(updated),
        )
        if search:
            return SearchItem(score=score, **kwargs)
        return Item(**kwargs)

    def _text_for(self, value: dict[str, Any], index: Any) -> str | None:
        if index is False:
            return None
        paths = [tokenize_path(f) for f in index] if isinstance(index, list) else self._field_paths
        if paths:
            parts: list[str] = []
            for p in paths:
                parts.extend(t for t in get_text_at_path(value, p) if t)
            return "\n".join(parts) or None
        for k in _DEFAULT_TEXT_KEYS:
            if isinstance(value.get(k), str) and value[k].strip():
                return value[k]
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    def _refcount(self, drawer_id: str) -> int:
        with self._db_lock:
            (n,) = self._db.execute(
                "SELECT COUNT(*) FROM items WHERE drawer_id = ?", (drawer_id,)
            ).fetchone()
        return n

    def _forget_if_unused(self, drawer_id: str | None) -> None:
        if drawer_id and self._refcount(drawer_id) == 0:
            try:
                self._client.call_tool("palace_delete_drawer", {"drawer_id": drawer_id})
            except PalaceError:
                pass

    def _file_drawer(self, namespace: tuple[str, ...], key: str, text: str) -> str | None:
        """Store text in Palace and return the drawer id (reusing a duplicate in our wing)."""
        source = "langgraph://" + "/".join(namespace) + "#" + key
        args = {
            "wing": self.wing,
            "room": _room_for(namespace),
            "content": text,
            "source_file": source,
            "added_by": "langgraph",
        }
        res = self._client.call_tool("palace_add_drawer", args)
        if res.get("success"):
            return res.get("drawer_id")
        if res.get("reason") == "duplicate":
            for m in res.get("matches", []):
                if m.get("wing") == self.wing:
                    return m.get("id")
            # Near-identical text lives in another wing (for example your coding
            # memory). File a copy here with a provenance line so it is distinct.
            args["content"] = f"[{source}]\n{text}"
            res = self._client.call_tool("palace_add_drawer", args)
            if res.get("success"):
                return res.get("drawer_id")
            for m in res.get("matches", []):
                if m.get("wing") == self.wing:
                    return m.get("id")
        raise PalaceError(f"palace_add_drawer failed: {res}")

    # -- ops -----------------------------------------------------------------
    def _get(self, op: GetOp) -> Item | None:
        with self._db_lock:
            row = self._db.execute(
                "SELECT ns, key, value, drawer_id, created_at, updated_at FROM items WHERE ns = ? AND key = ?",
                (_ns_key(op.namespace), op.key),
            ).fetchone()
        return self._row_to_item(row) if row else None

    def _put(self, op: PutOp) -> None:
        ns = _ns_key(op.namespace)
        with self._db_lock:
            existing = self._db.execute(
                "SELECT drawer_id, created_at FROM items WHERE ns = ? AND key = ?", (ns, op.key)
            ).fetchone()
        old_drawer = existing[0] if existing else None

        if op.value is None:  # delete
            with self._db_lock, self._db:
                self._db.execute("DELETE FROM items WHERE ns = ? AND key = ?", (ns, op.key))
            self._forget_if_unused(old_drawer)
            return

        now = _now().isoformat()
        created = existing[1] if existing else now
        text = self._text_for(op.value, op.index)
        drawer_id: str | None = None
        if text is not None:
            if old_drawer and self._refcount(old_drawer) == 1:
                res = self._client.call_tool(
                    "palace_update_drawer", {"drawer_id": old_drawer, "content": text}
                )
                drawer_id = old_drawer if res.get("success") else None
            if drawer_id is None:
                drawer_id = self._file_drawer(op.namespace, op.key, text)
        with self._db_lock, self._db:
            self._db.execute(
                """INSERT INTO items (ns, key, value, drawer_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(ns, key) DO UPDATE SET value = excluded.value,
                     drawer_id = excluded.drawer_id, updated_at = excluded.updated_at""",
                (ns, op.key, json.dumps(op.value, ensure_ascii=False), drawer_id, created, now),
            )
        if old_drawer and old_drawer != drawer_id:
            self._forget_if_unused(old_drawer)

    def _prefix_rows(self, prefix: tuple[str, ...]) -> list[tuple]:
        with self._db_lock:
            rows = self._db.execute(
                "SELECT ns, key, value, drawer_id, created_at, updated_at FROM items ORDER BY updated_at DESC"
            ).fetchall()
        return [r for r in rows if _ns_from_key(r[0])[: len(prefix)] == tuple(prefix)]

    def _search(self, op: SearchOp) -> list[SearchItem]:
        prefix = tuple(op.namespace_prefix)
        if not op.query:
            rows = [r for r in self._prefix_rows(prefix) if _matches_filter(json.loads(r[2]), op.filter)]
            return [self._row_to_item(r, search=True) for r in rows[op.offset : op.offset + op.limit]]

        want = op.offset + op.limit
        res = self._client.call_tool(
            "palace_search",
            {
                "query": op.query,
                "wing": self.wing,
                "limit": max(want * self.oversample, 10),
                "context": False,
            },
        )
        hits = res.get("results", [])
        by_drawer: dict[str, list[tuple]] = {}
        for r in self._prefix_rows(prefix):
            if r[3]:
                by_drawer.setdefault(r[3], []).append(r)
        out: list[SearchItem] = []
        seen: set[tuple[str, str]] = set()
        for hit in hits:
            for row in by_drawer.get(hit.get("id", ""), []):
                if (row[0], row[1]) in seen or not _matches_filter(json.loads(row[2]), op.filter):
                    continue
                seen.add((row[0], row[1]))
                out.append(self._row_to_item(row, score=hit.get("similarity"), search=True))
        return out[op.offset : want]

    def _list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        with self._db_lock:
            keys = [r[0] for r in self._db.execute("SELECT DISTINCT ns FROM items").fetchall()]
        spaces = {_ns_from_key(k) for k in keys}
        if op.match_conditions:
            spaces = {s for s in spaces if all(_ns_matches(s, c) for c in op.match_conditions)}
        if op.max_depth is not None:
            spaces = {s[: op.max_depth] for s in spaces}
        ordered = sorted(spaces)
        return ordered[op.offset : op.offset + op.limit]
