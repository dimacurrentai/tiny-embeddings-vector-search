# Mini vector search

A local MCP server for small text collections, using exactly
`sentence-transformers/all-MiniLM-L6-v2`. It runs on CPU with ONNX Runtime;
no GPU, API key, PyTorch, Ollama daemon, or vector database is required.
Four direct dependencies: `mcp`, `numpy`, `onnxruntime`, and `tokenizers`.
SQLite and the test runner come with Python.

## Tests

```sh
uv run python -m unittest discover -s tests -v
```
