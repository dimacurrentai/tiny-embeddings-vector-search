import unittest
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from http_server import running_server

from temp_paths import temporary_directory


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_discovery_and_query(self):
        with temporary_directory() as temporary:
            with running_server(Path(temporary)) as url:
                async with streamable_http_client(url) as (read, write, _):
                    async with ClientSession(read, write) as client:
                        await client.initialize()
                        tools = await client.list_tools()
                        self.assertIn("search", {tool.name for tool in tools.tools})
                        result = await client.call_tool("search", {"query": "hello"})
                        self.assertFalse(result.isError)
                        self.assertEqual(result.structuredContent, {"result": []})
