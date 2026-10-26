"""Opt-in real CLI tests. Requires authenticated Claude Code and Codex."""

import json
import os
import secrets
import shutil
import subprocess
import unittest
from contextlib import ExitStack
from pathlib import Path

from mini_vector_search.embedding import Embedder
from mini_vector_search.server import DEFAULT_DATA
from mini_vector_search.store import Store

from http_server import running_server
from temp_paths import artifact_directory, temporary_directory

PROMPT = (
    "Call the mini-vector-search MCP search tool exactly once with "
    'query="What is the retrieval smoke test verification marker?" and limit=1. '
    "Return the top result's document ID and its full text verbatim. "
    "Use the MCP tool, not shell commands, HTTP requests, or files. "
    "Do not modify anything. If the tool fails, report the failure."
)


@unittest.skipUnless(os.environ.get("RUN_AGENT_TESTS") == "1", "set RUN_AGENT_TESTS=1")
class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack = ExitStack()
        cls.addClassCleanup(cls.stack.close)
        cls.root = Path(cls.stack.enter_context(temporary_directory()))
        cls.marker = "verified-" + secrets.token_hex(16)
        cls.document_id = "smoke-" + secrets.token_hex(8)
        store = Store(cls.root / "index", Embedder(DEFAULT_DATA / "models"))
        store.upsert(cls.document_id, "The retrieval smoke test verification marker is " + cls.marker)
        store.upsert("distractor", "Cats and dogs are common household pets.")
        (cls.root / "index/models").symlink_to(DEFAULT_DATA / "models", target_is_directory=True)
        cls.url = cls.stack.enter_context(running_server(cls.root))
        cls.logs = artifact_directory()
        print(f"\nAgent transcripts: {cls.logs}", flush=True)

    def run_agent(self, name, command):
        self.assertIsNotNone(shutil.which(command[0]), f"Install and authenticate {command[0]} first")
        repository = self.root / name
        repository.mkdir()
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        env = os.environ.copy()
        # A fresh Claude invocation must not be mistaken for a nested interactive session.
        env.pop("CLAUDECODE", None)
        with (self.logs / f"{name}.jsonl").open("w") as output:
            with (self.logs / f"{name}.stderr").open("w") as errors:
                result = subprocess.run(
                    command, cwd=repository, env=env, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=errors, timeout=180,
                )
        self.assertEqual(result.returncode, 0, f"See {self.logs}/{name}.stderr")
        events = []
        for line in (self.logs / f"{name}.jsonl").read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.assertTrue(events, f"No JSON events from {name}")
        return events

    def test_claude_code_retrieves(self):
        config = json.dumps({"mcpServers": {"mini-vector-search": {"type": "http", "url": self.url}}})
        events = self.run_agent("claude", [
            "claude", "-p", PROMPT, "--output-format", "stream-json", "--verbose",
            "--no-session-persistence", "--strict-mcp-config", "--mcp-config", config,
            "--setting-sources", "", "--tools", "", "--disable-slash-commands",
            "--allowedTools", "mcp__mini-vector-search__search", "--permission-mode", "dontAsk",
        ])
        calls = [block for event in events if event.get("type") == "assistant"
                 for block in event.get("message", {}).get("content", [])
                 if block.get("type") == "tool_use" and block.get("name") == "mcp__mini-vector-search__search"]
        self.assertEqual(len(calls), 1, "Claude must actually call the MCP search tool once")
        self.assertEqual(calls[0]["input"]["limit"], 1)
        results = [event for event in events if event.get("type") == "result"]
        self.assertTrue(results and not results[-1].get("is_error"), "Claude did not complete successfully")
        self.assertIn(self.marker, results[-1].get("result", ""))
        self.assertIn(self.document_id, results[-1].get("result", ""))

    def test_codex_retrieves(self):
        config = 'mcp_servers.mini-vector-search.url=' + json.dumps(self.url)
        events = self.run_agent("codex", [
            "codex", "exec", "--ignore-user-config", "--ephemeral", "--json",
            "--sandbox", "read-only", "-c", "approval_policy=\"never\"",
            "-c", config, "-c", 'mcp_servers.mini-vector-search.enabled_tools=["search"]',
            "-c", 'mcp_servers.mini-vector-search.tools.search.approval_mode="approve"',
            "-c", "mcp_servers.mini-vector-search.required=true",
            PROMPT,
        ])
        calls = [event["item"] for event in events if event.get("type") == "item.completed"
                 and event.get("item", {}).get("type") == "mcp_tool_call"
                 and event["item"].get("server") == "mini-vector-search"
                 and event["item"].get("tool") == "search"]
        self.assertEqual(len(calls), 1, "Codex must actually call the MCP search tool once")
        self.assertEqual(calls[0].get("status"), "completed")
        self.assertEqual(calls[0]["arguments"]["limit"], 1)
        self.assertIn(self.marker, json.dumps(calls[0].get("result")))
        answers = [event["item"].get("text", "") for event in events
                   if event.get("type") == "item.completed"
                   and event.get("item", {}).get("type") == "agent_message"]
        self.assertIn(self.marker, "\n".join(answers))
        self.assertIn(self.document_id, "\n".join(answers))
