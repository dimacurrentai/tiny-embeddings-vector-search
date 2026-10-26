import os
import sys
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from temp_paths import temporary_directory


class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdio_discovery_and_validation(self):
        with temporary_directory() as directory:
            parameters = StdioServerParameters(
                command=sys.executable,
                args=["-m", "mini_vector_search.server", "--data-dir", directory],
            )
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    tools = await client.list_tools()
                    self.assertEqual(
                        {tool.name for tool in tools.tools},
                        {"upsert_document", "search", "delete_document", "stats"},
                    )
                    result = await client.call_tool("stats", {})
                    self.assertFalse(result.isError)
                    self.assertEqual(result.structuredContent["count"], 0)
                    result = await client.call_tool("search", {"query": "hello"})
                    self.assertFalse(result.isError)
                    result = await client.call_tool("upsert_document", {"id": "x", "text": " "})
                    self.assertTrue(result.isError)
                    result = await client.call_tool("delete_document", {"id": "missing"})
                    self.assertEqual(result.structuredContent, {"deleted": False})
            self.assertFalse((Path(directory) / "models").exists())

    @unittest.skipUnless(os.environ.get("RUN_MODEL_TESTS") == "1", "set RUN_MODEL_TESTS=1 for real-model test")
    async def test_real_model_over_stdio(self):
        # Reuse the ignored model cache but keep test documents out of the user index.
        from mini_vector_search.embedding import Embedder
        from mini_vector_search.server import DEFAULT_DATA
        import numpy as np

        embedder = Embedder(DEFAULT_DATA / "models")
        vectors = embedder.encode(["A kitten sleeps.", "SQL database indexes."])
        self.assertEqual(vectors.shape, (2, 384))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
        # Padding must not change the short sentence's embedding.
        np.testing.assert_allclose(
            vectors[0], embedder.encode(["A kitten sleeps."])[0], atol=1e-5
        )
        with temporary_directory() as directory:
            (Path(directory) / "models").symlink_to(DEFAULT_DATA / "models", target_is_directory=True)
            parameters = StdioServerParameters(
                command=sys.executable,
                args=["-m", "mini_vector_search.server", "--data-dir", directory],
            )
            async with stdio_client(parameters) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    for id, text in [("pets", "Cats and dogs are common household pets."),
                                     ("sql", "Create a database index to speed up SQL queries.")]:
                        result = await client.call_tool("upsert_document", {"id": id, "text": text})
                        self.assertFalse(result.isError)
                    result = await client.call_tool("search", {"query": "How do I make database queries faster?", "limit": 1})
                    self.assertFalse(result.isError)
                    self.assertEqual(result.structuredContent["result"][0]["id"], "sql")
                    result = await client.call_tool("delete_document", {"id": "sql"})
                    self.assertTrue(result.structuredContent["deleted"])
