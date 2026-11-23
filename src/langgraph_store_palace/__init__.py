"""LangGraph long-term memory on Palace (palace-rs): local, verbatim, hybrid search.

Not associated in any way with the popular mempalace python.
"""

from ._mcp import PalaceError, PalaceMCPClient
from .store import PalaceStore

__all__ = ["PalaceStore", "PalaceMCPClient", "PalaceError"]
__version__ = "0.1.0"
