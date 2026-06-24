"""
mcp_client.py - the Python side that talks to mcp_server.py.

Starts the server as a background process and lets you call its tools
(uniprot_search, pdb_search, alphafold_lookup) as normal Python methods. One server
process is reused for all calls.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
DEFAULT_SERVER = HERE / "mcp_server.py"


class BioMCPClient:
    def __init__(self, server_path: Optional[Path] = None, python_exe: Optional[str] = None):
        self.server_path = str(server_path or DEFAULT_SERVER)
        self.python_exe = python_exe or sys.executable
        self.proc: Optional[subprocess.Popen] = None
        self._id = 0

    def start(self):
        self.proc = subprocess.Popen(
            [self.python_exe, "-u", self.server_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", bufsize=1,
        )
        self._rpc("initialize", {"protocolVersion": "2024-11-05",
                                 "capabilities": {}, "clientInfo": {"name": "kb-secretion-swarm-client"}})
        self._notify("notifications/initialized")
        return self

    def stop(self):
        if self.proc:
            try:
                self.proc.stdin.close()
                self.proc.terminate()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
            self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- transport
    def _send(self, obj: Dict[str, Any]):
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _notify(self, method: str, params: Optional[dict] = None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _rpc(self, method: str, params: Optional[dict] = None) -> Dict[str, Any]:
        self._id += 1
        mid = self._id
        self._send({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed unexpectedly")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == mid:
                if "error" in msg:
                    raise RuntimeError(f"MCP error: {msg['error']}")
                return msg.get("result", {})

    # -- tools
    def call(self, name: str, arguments: Dict[str, Any]) -> Any:
        if self.proc is None:
            self.start()
        result = self._rpc("tools/call", {"name": name, "arguments": arguments})
        content = (result or {}).get("content") or []
        if not content:
            return None
        text = content[0].get("text", "")
        if result.get("isError"):
            raise RuntimeError(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    def uniprot_search(self, query: str, size: int = 5, taxonomy_id: int = 5820) -> List[dict]:
        return self.call("uniprot_search",
                         {"query": query, "size": size, "taxonomy_id": taxonomy_id}) or []

    def pdb_search(self, query: str, size: int = 5) -> List[dict]:
        return self.call("pdb_search", {"query": query, "size": size}) or []

    def alphafold_lookup(self, accession: str) -> dict:
        return self.call("alphafold_lookup", {"accession": accession}) or {}


if __name__ == "__main__":
    with BioMCPClient() as mcp:
        print("tools:", [t["name"] for t in mcp._rpc("tools/list").get("tools", [])])
        hits = mcp.uniprot_search("PfCRT", size=2)
        print("PfCRT ->", json.dumps(hits, indent=2)[:700])
