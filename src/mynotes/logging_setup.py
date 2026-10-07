"""Logging setup with secret masking for all mynotes entry points.

:func:`configure_logging` installs one stderr handler on the root logger.
A :class:`SecretMaskingFilter` on that handler rewrites any occurrence of the
secret values from the current settings (API keys, Postgres DSN, OpenSearch
password, passwords embedded in the DSN) into their masked form, so real
secrets never reach logs. Calling :func:`configure_logging` again replaces the
previous handler instead of duplicating it.

When the ``MYNOTES_LOG_FILE`` environment variable is set (the Windows
scheduled task sets it, PLAN.md §7.8), a ``RotatingFileHandler`` (10 MB x 5
files) with the same formatter and masking filter is added, so headless runs
keep a log file; a file that cannot be opened is skipped with a warning.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from typing import Any
from urllib.parse import urlparse

from mynotes.config import Settings, get_settings, mask_secret

_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_HANDLER_NAME = "mynotes"
_LOG_FILE_VAR = "MYNOTES_LOG_FILE"
_LOG_FILE_MAX_BYTES = 10 * 1024 * 1024
_LOG_FILE_BACKUPS = 5
_MIN_SECRET_LENGTH = 6
_QUIET_LOGGERS: tuple[str, ...] = ("httpx", "httpcore", "urllib3", "opensearch")

_STANDARD_RECORD_ATTRS: frozenset[str] = frozenset(
    vars(logging.LogRecord("n", logging.INFO, "p", 1, "m", None, None))
) | {"message", "asctime", "taskName"}


class JsonFormatter(logging.Formatter):
    """Format each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` with ``ts``/``level``/``logger``/``msg`` plus extras.

        Args:
            record: The log record to render.

        Returns:
            A single-line JSON string; non-serializable values fall back to
            ``str()``.
        """
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in _STANDARD_RECORD_ATTRS
            }
        )
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class SecretMaskingFilter(logging.Filter):
    """Replace known secret values in record messages with masked forms."""

    def __init__(self, secrets: list[str]) -> None:
        """Store the replacements, longest secret first.

        Args:
            secrets: Secret values to mask; empties and values shorter than
                6 characters are ignored.
        """
        super().__init__()
        ordered = sorted(
            {s for s in secrets if len(s) >= _MIN_SECRET_LENGTH}, key=len, reverse=True
        )
        self._replacements = [(secret, mask_secret(secret)) for secret in ordered]

    def filter(self, record: logging.LogRecord) -> bool:
        """Mask every secret occurrence in the record message.

        Args:
            record: The log record to rewrite in place.

        Returns:
            Always True so the record keeps flowing to the handler.
        """
        message = record.getMessage()
        for secret, masked in self._replacements:
            if secret in message:
                message = message.replace(secret, masked)
        record.msg = message
        record.args = None
        return True


def _collect_secrets(settings: Settings) -> list[str]:
    """Gather the current secret values worth masking.

    Args:
        settings: The settings to read secrets from.

    Returns:
        API keys for the model endpoints, the Postgres DSN, any password
        embedded in that DSN, and the OpenSearch password. Empty values are
        skipped; the length filter happens in :class:`SecretMaskingFilter`.
    """
    candidates: list[str] = []
    for endpoint in (
        settings.models.embedding,
        settings.models.reranker,
        settings.models.llm,
        settings.models.extract_llm,
    ):
        if endpoint is not None and endpoint.api_key is not None:
            candidates.append(endpoint.api_key.get_secret_value())
    if settings.postgres.dsn is not None:
        dsn = settings.postgres.dsn.get_secret_value()
        candidates.append(dsn)
        try:
            password = urlparse(dsn).password
        except ValueError:
            password = None
        if password:
            candidates.append(password)
    if settings.opensearch.password is not None:
        candidates.append(settings.opensearch.password.get_secret_value())
    return candidates


def _file_handler(log_path: str, formatter: logging.Formatter) -> RotatingFileHandler | None:
    """Build the rotating file handler for ``MYNOTES_LOG_FILE``.

    Args:
        log_path: Target file from the environment.
        formatter: The formatter shared with the stderr handler.

    Returns:
        A handler with the masking filter installed, or None when the file
        cannot be opened (the warning is logged by the caller).
    """
    try:
        handler = RotatingFileHandler(
            log_path, maxBytes=_LOG_FILE_MAX_BYTES, backupCount=_LOG_FILE_BACKUPS
        )
    except OSError:
        return None
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(formatter)
    handler.addFilter(SecretMaskingFilter(_collect_secrets(get_settings())))
    return handler


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    """Configure the root logger with a masked stderr handler.

    Args:
        level: Root logger level name (case-insensitive, e.g. ``"INFO"``).
        json: Emit one JSON object per record instead of plain text.

    Raises:
        ValueError: If ``level`` is not a known logging level name.
    """
    numeric = logging.getLevelNamesMapping().get(level.upper())
    if numeric is None:
        raise ValueError(f"unknown logging level: {level!r}")

    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == _HANDLER_NAME:
            root.removeHandler(handler)
            handler.close()

    formatter = JsonFormatter() if json else logging.Formatter(_TEXT_FORMAT)
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(formatter)
    handler.addFilter(SecretMaskingFilter(_collect_secrets(get_settings())))
    root.addHandler(handler)
    root.setLevel(numeric)

    log_path = os.environ.get(_LOG_FILE_VAR)
    if log_path:
        file_handler = _file_handler(log_path, formatter)
        if file_handler is None:
            logging.getLogger(__name__).warning(
                "could not open log file %s; file logging is disabled", log_path
            )
        else:
            root.addHandler(file_handler)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
