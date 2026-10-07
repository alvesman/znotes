"""Allow ``python -m mynotes`` (the Windows scheduled task runs pythonw -m mynotes)."""

from mynotes.cli import app

app()
