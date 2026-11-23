"""Minimal JSON-RPC client for the `palace mcp` stdio server.

Palace speaks MCP over stdio, one JSON object per line. This client keeps one
server process alive and serialises calls with a lock, so a store instance can
be shared across threads.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from typing import Any


class PalaceError(RuntimeError):
    """Raised when the palace server returns an error or cannot be reached."""


class PalaceMCPClient:
    def __init__(
        self,
        palace_bin: str | None = None,
        *,
        palace_path: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        resolved = palace_bin or os.environ.get("PALACE_BIN") or shutil.which("palace")
        if not resolved:
            raise PalaceError(
                "Could not find the `palace` binary. Install it from "
                "https://github.com/AncientiCe/palace-rs or pass palace_bin=..."
            )
        self.palace_bin = resolved
        self._env = dict(os.environ)
        if env:
            self._env.update(env)
        if palace_path:
            # Keeps LangGraph app memory in its own palace.db if wanted.
            self._env["PALACE_PALACE_PATH"] = palace_path
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._next_id = 0

    # -- lifecycle ---------------------------------------------------------
    def _start(self) -> None:
        self._proc = subprocess.Popen(
            [self.palace_bin, "mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=self._env,
            bufsize=1,
        )
        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "langgraph-store-palace", "version": "0.1.0"},
            },
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    def close(self) -> None:
        with self._lock:
            if self._proc and self._proc.poll() is None:
                try:
                    self._proc.stdin.close()  # type: ignore[union-attr]
                    self._proc.wait(timeout=5)
                except Exception:
                    self._proc.kill()
            self._proc = None

    # -- transport ---------------------------------------------------------
    def _send(self, msg: dict[str, Any]) -> None:
        assert self._proc and self._proc.stdin
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._next_id += 1
        req_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        assert self._proc and self._proc.stdout
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise PalaceError("palace mcp exited unexpectedly")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") != req_id:
                continue  # notifications or stale replies
            if "error" in msg:
                raise PalaceError(str(msg["error"]))
            return msg.get("result", {})

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one Palace MCP tool and return its parsed JSON payload."""
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._start()
            result = self._request("tools/call", {"name": name, "arguments": arguments})
        texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
        raw = "\n".join(texts).strip()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"text": raw}
        if result.get("isError"):
            raise PalaceError(raw or f"{name} failed")
        return payload
