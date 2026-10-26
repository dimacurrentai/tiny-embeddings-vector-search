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
