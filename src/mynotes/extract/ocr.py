"""Optional OCR fallback for image-only PDFs (PLAN.md §7.2).

Runs ``ocrmypdf`` (which drives Tesseract) as a subprocess; both binaries must
be on the PATH. ``--output-type pdf`` avoids the PDF/A conversion, which needs
Ghostscript. The OCR'd copy lands in a temporary directory: on success the
caller deletes it, on failure this module cleans up itself.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGES = "por+eng"
DEFAULT_TIMEOUT_S = 600


def ocr_available() -> bool:
    """Return True when both ``ocrmypdf`` and ``tesseract`` are on the PATH."""
    return shutil.which("ocrmypdf") is not None and shutil.which("tesseract") is not None


def ocr_pdf(
    path: Path, languages: str = DEFAULT_LANGUAGES, timeout_s: int = DEFAULT_TIMEOUT_S
) -> Path:
    """OCR one PDF into a new file inside a temporary directory.

    Args:
        path: The (usually image-only) PDF to OCR.
        languages: Tesseract language string passed with ``-l``.
        timeout_s: Subprocess timeout in seconds.

    Returns:
        Path of the OCR'd PDF in a fresh temporary directory; the caller
        deletes it when done.

    Raises:
        FileNotFoundError: When ocrmypdf or tesseract is not installed.
        subprocess.TimeoutExpired: When OCR exceeds ``timeout_s``.
        RuntimeError: When ocrmypdf exits non-zero (message carries the last
            line of its stderr).
    """
    if not ocr_available():
        raise FileNotFoundError("ocrmypdf and tesseract must be installed for OCR")
    output_dir = Path(tempfile.mkdtemp(prefix="mynotes-ocr-"))
    output = output_dir / f"{path.stem}-ocr.pdf"
    command = [
        "ocrmypdf",
        "--skip-text",
        "--output-type",
        "pdf",
        "-l",
        languages,
        str(path),
        str(output),
    ]
    logger.info("OCR: %s", " ".join(command))
    try:
        result = subprocess.run(command, timeout=timeout_s, capture_output=True, text=True)
    except BaseException:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise
    if result.returncode != 0:
        shutil.rmtree(output_dir, ignore_errors=True)
        lines = (result.stderr or result.stdout or "").strip().splitlines()
        detail = lines[-1] if lines else "no output"
        raise RuntimeError(f"ocrmypdf exited with status {result.returncode}: {detail}")
    if not output.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
        raise RuntimeError("ocrmypdf reported success but produced no output file")
    return output
