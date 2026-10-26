# Mini vector search

A local MCP server for small text collections, using exactly
`sentence-transformers/all-MiniLM-L6-v2`. It runs on CPU with ONNX Runtime;
no GPU, API key, PyTorch, Ollama daemon, or vector database is required.
Four direct dependencies: `mcp`, `numpy`, `onnxruntime`, and `tokenizers`.
SQLite and the test runner come with Python.

## Run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from this checkout:

```sh
uv run mini-vector-search
```

uv installs Python 3.12 and dependencies automatically. The process waits for
an MCP client on stdin. An optional localhost HTTP transport is described below.
Supported targets are macOS, Linux, and WSL. In WSL, install and run uv and
the MCP client inside the Linux environment.

The first document insertion downloads the model (roughly 90 MB) from Hugging
Face. Files are pinned to a model revision, verified against SHA-256 checksums,
and stored in `.data/models/`.
Documents and vectors persist in `.data/documents.sqlite3`. The entire `.data/`
directory is gitignored. Downloads require internet; after caching, inference
and search run offline. Failed downloads can be retried by repeating the tool call.
Discovery, stats, deletion, and empty searches do not load the model.

Paths default to this checkout, independent of the client's working directory.
An optional `--data-dir /absolute/path` selects another index and model cache.
Run from the checkout with uv; a standalone wheel installation instead places
the default data beside the installed package and may require `--data-dir`.

## Connect a local coding agent

Add this entry to the agent's MCP server configuration, substituting the
absolute checkout path (and the absolute uv executable path if necessary):

```json
{
  "mcpServers": {
    "mini-vector-search": {
      "command": "uv",
      "args": [
        "run", "--directory", "/absolute/path/to/mini-vector-search",
        "mini-vector-search"
      ]
    }
  }
}
```

The client's one-time server registration is the only required configuration.
Use a generous first-call timeout (e.g. 180 seconds) for the initial download,
or run the real-model test below first to populate the cache.

| Tool | Arguments | Result |
| --- | --- | --- |
| `upsert_document` | `id`, `text` | Added/replaced ID |
| `search` | `query`, optional `limit` (default 5; 1–100) | IDs, original text, cosine scores, highest first |
| `delete_document` | `id` | Whether a document was deleted |
| `stats` | none | Document count, model, revision, dimensions |

Example: ask your agent to insert `{"id":"db-indexes","text":"Database indexes speed up SQL queries."}`
with `upsert_document`, then call `search` with `{"query":"How can I make database lookups faster?"}`.

Each insertion replaces the same ID atomically. Text is stored in full, but
the model embeds only the first 256 tokens, so insert short passages as separate
IDs. There is no automatic file scanning or chunking. The 384-dimensional
embeddings use attention-mask-aware mean pooling and L2 normalization, as
specified in the [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
Search is an exact linear scan, suitable for small collections; all vectors
are loaded into memory per query. Scores are similarities, not probabilities.
Logs go to stderr, leaving stdout for MCP.

## Share a localhost MCP server between agents

```sh
uv run mini-vector-search --transport streamable-http --port 8000
```

The endpoint is `http://127.0.0.1:8000/mcp`. It binds only to loopback and has
no authentication; use it on your own machine. Stop it with Ctrl-C.

Claude Code project `.mcp.json`:

```json
{
  "mcpServers": {
    "mini-vector-search": {
      "type": "http",
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Codex project `.codex/config.toml` (in a trusted project):

```toml
[mcp_servers.mini-vector-search]
url = "http://127.0.0.1:8000/mcp"
enabled_tools = ["search", "stats", "upsert_document", "delete_document"]

[mcp_servers.mini-vector-search.tools.search]
approval_mode = "approve"
```

The search override permits noninteractive queries; other tools retain the
client's default approval behavior. See the official
[Codex MCP configuration](https://developers.openai.com/codex/mcp) and
[Claude Code MCP documentation](https://code.claude.com/docs/en/mcp).

## Tests

```sh
uv run python -m unittest discover -s tests -v
RUN_MODEL_TESTS=1 uv run python -m unittest discover -s tests -v
```

The default suite runs offline after dependency installation. It covers
ranking, replacement, persistence, deletion, validation, download cleanup, cache
reuse, checksum verification, and actual MCP subprocess discovery and calls.
The second command also downloads and executes the real model, checks normalized
embeddings and padding, and exercises semantic search through MCP.
Both stdio and localhost HTTP are covered. Tests reuse `.data/models/` and create
scratch directories under gitignored `tmp/test-<random>/`, removing them afterward.
Neither command modifies your document index.
