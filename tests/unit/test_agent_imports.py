"""The agent runs on the core dependencies alone (the agent image has no `server` extra)."""

from __future__ import annotations

import json
import subprocess
import sys

_SERVER_PACKAGES = (
    "sqlalchemy",
    "psycopg",
    "alembic",
    "opensearchpy",
    "fastapi",
    "uvicorn",
    "pymupdf",
    "markitdown",
    "tokenizers",
    "tiktoken",
    "networkx",
    "numpy",
    "fast_langdetect",
    "jinja2",
    "jwt",
    "mcp",
)

_PROBE = """
import json, sys
from typer.testing import CliRunner
from mynotes.cli import app
assert CliRunner().invoke(app, ["agent", "--help"]).exit_code == 0
import mynotes.agent.client, mynotes.agent.hashcache, mynotes.agent.runner, mynotes.agent.sync
import mynotes.scan.hashing, mynotes.scan.ignore, mynotes.scan.walker, mynotes.scan.watch
import mynotes.platform.fsinfo
print(json.dumps(sorted(m.split(".")[0] for m in sys.modules)))
"""


def test_agent_path_imports_no_server_package() -> None:
    """Importing the agent commands and modules loads none of the server-only packages.

    A fresh interpreter, so modules imported by other tests do not count.
    """
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True, timeout=120
    )
    loaded = set(json.loads(result.stdout.strip().splitlines()[-1]))
    leaked = sorted(package for package in _SERVER_PACKAGES if package in loaded)
    assert leaked == [], f"the agent path imports server packages: {leaked}"
