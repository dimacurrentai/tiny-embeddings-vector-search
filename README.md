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
embeddings and padding, exercises semantic search through MCP, and runs the CLINC150 subset evaluation.
Both stdio and localhost HTTP are covered. Tests reuse `.data/models/` and create
scratch directories under gitignored `tmp/test-<random>/`, removing them afterward.
Neither command modifies your document index.

### Small, vendored quality evaluation

[tests/fixtures/clinc150](tests/fixtures/clinc150) contains 200 human-labeled
snippets from 20 intents, in readable JSON (~24 KB including IDs and field names).
The original CC BY 3.0 license, full attribution, pinned upstream revision,
source checksum, and deterministic reproduction script are included. No dataset
download or extra dependency is needed to run it.

Each snippet queries the other 199 through the real Store search path. Matching
intent is the relevance label; the query itself is excluded. This measures
intent retrieval on a fixed subset, not general embedding quality or the full
CLINC150 benchmark. Selection was fixed before measuring performance.

```sh
RUN_MODEL_TESTS=1 uv run python -m unittest discover -s tests -p test_quality.py -v
```

Measured on macOS: top-1 accuracy **98%**, hit rate at 5 **100%**, recall at 5
**51.28%**, MRR at 5 **0.9877**. Recall at 5 has a maximum of 5/9 (55.56%) because
each query has nine relevant snippets. The regression floors, set before the
first run, are 70% top-1 and 85% hit rate at 5. They are guards against regressions,
not estimates for unseen queries.

### Real Claude Code and Codex MCP smoke tests

These are two separate opt-in tests using installed, authenticated CLIs and their
configured/default models. They make real model requests and use your accounts'
quota. No credentials or global client configuration are changed. Tested with
Claude Code 2.1.308 and Codex CLI 0.164.0; older CLIs may lack the flags used here.

```sh
# Both agent tests against one temporary localhost server:
RUN_AGENT_TESTS=1 uv run python -m unittest discover -s tests -p test_agents.py -v

# Run either separately:
RUN_AGENT_TESTS=1 uv run python -m unittest discover -s tests -p test_agents.py -k claude -v
RUN_AGENT_TESTS=1 uv run python -m unittest discover -s tests -p test_agents.py -k codex -v
```

Each agent runs in its own temporary Git repository under `tmp/test-<random>/`.
The tests seed a temporary
index with a fresh random verification marker and a distractor, start the real
service on a free loopback port, and request exactly one MCP search. Assertions
check the actual tool-call event and the retrieved marker and document ID in the
answer. The marker is never included in the prompt. Claude allows the search tool;
Codex exposes and preapproves only search for its invocation.

Server processes and temporary repositories/indexes are cleaned up even on test
failure. Agent JSON event transcripts and stderr remain in gitignored
`tmp/agent-smoke-<random>/` for inspection; each run prints its directory and keeps
its own transcripts. Agent tests
are skipped in normal CI; HTTP protocol and model/quality tests run in CI.

### Pure C embedder comparison

[bench/embed.c](bench/embed.c) computes the same embedding with no
dependencies: a WordPiece tokenizer, the six BERT layers, mean pooling, and
L2 normalization in about 400 lines of single-threaded float32 C. It exists to
show what the model does and to measure ONNX Runtime against plain C.

```sh
uv run python bench/export_weights.py   # once: tensors and vocabulary from the cached ONNX model
uv run python bench/compare.py          # builds bench/build/embed, checks agreement, times both
```

The exporter decodes the ONNX protobuf directly, so no extra package is needed.
Both runners speak the same line protocol: one JSON string in, one line out with
the token IDs and the final 384-dimensional vector. The comparison sends the 200
CLINC150 snippets plus mixed-case, punctuated, and accented sentences through the
C embedder, ONNX Runtime with default threads, and ONNX Runtime pinned to one
thread, then times each after its first reply so model load is excluded.

Measured on an Apple M3 Pro, best of 5 rounds, 12.9 tokens per sentence on average:

| Runner | Per sentence | Sentences/s |
| --- | --- | --- |
| ONNX Runtime, default threads | 1.7 ms | 580 |
| ONNX Runtime, one thread | 3.9 ms | 260 |
| C, one thread | 5.9 ms | 170 |

Token IDs are identical for every sentence, and vectors agree to within 3e-7,
which is float32 rounding from different summation orders. ONNX Runtime wins
with hand-tuned SIMD matrix kernels and threading; the C keeps the sums in
model order and only unrolls the matrix loops. The C tokenizer matches the
reference exactly for ASCII, Latin-1, and Latin Extended-A text; other scripts
pass through to WordPiece without Unicode case folding.

`RUN_MODEL_TESTS=1` also runs [tests/test_c_embedding.py](tests/test_c_embedding.py),
which builds the C embedder and asserts identical token IDs and vectors within
1e-5 on representative sentences, including truncation at 256 tokens.
