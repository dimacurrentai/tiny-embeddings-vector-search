"""Local MCP entry point; stdout belongs exclusively to MCP."""

import argparse
import logging
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from .store import Store

DEFAULT_DATA = Path(__file__).resolve().parent.parent / ".data"


def create_server(directory: Path = DEFAULT_DATA) -> FastMCP:
    store = Store(directory)
    server = FastMCP("mini-vector-search")

    @server.tool()
    def upsert_document(id: str, text: str) -> dict[str, Any]:
        """Add or replace a document by ID. Keep passages under 256 model tokens."""
        return store.upsert(id, text)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def search(query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find the nearest documents by cosine similarity; limit is 1 through 100."""
        return store.search(query, limit)

    @server.tool()
    def delete_document(id: str) -> dict[str, Any]:
        """Delete a document by ID, returning whether it existed."""
        return store.delete(id)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def stats() -> dict[str, Any]:
        """Return document count and model details without downloading the model."""
        return store.stats()

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    create_server(args.data_dir.resolve()).run(transport="stdio")


if __name__ == "__main__":
    main()
