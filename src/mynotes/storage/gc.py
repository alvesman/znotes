"""Garbage collection of the blob store: drop blobs no row references any more.

Runs with the nightly ``clean_caches`` job. A blob is live while a
``document_version`` or ``file`` row of its root carries its hash; after
``purge_tombstones`` removed a deleted document, its blobs become orphans and
are removed here. Partial uploads older than a day are swept as well.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from mynotes.db.models import Document, DocumentVersion, File
from mynotes.storage.blobs import BlobStore

logger = logging.getLogger(__name__)


def live_blobs(session: Session, root_id: int) -> set[str]:
    """The ``<hash>.<format>`` names still referenced for one root."""
    names: set[str] = set()
    versions = session.execute(
        select(DocumentVersion.content_hash, Document.format)
        .join(Document, Document.id == DocumentVersion.document_id)
        .where(Document.source_root_id == root_id, DocumentVersion.content_hash.is_not(None))
    ).all()
    names.update(f"{digest}.{fmt}" for digest, fmt in versions if digest)
    files = session.execute(
        select(File.content_hash, File.format).where(
            File.source_root_id == root_id, File.content_hash.is_not(None)
        )
    ).all()
    names.update(f"{digest}.{fmt}" for digest, fmt in files if digest and fmt)
    return names


def collect_garbage(session: Session, store: BlobStore, *, dry_run: bool = False) -> dict[str, Any]:
    """Remove unreferenced blobs and stale partial uploads.

    Returns:
        ``{"blobs_removed", "bytes_freed", "tmp_removed", "roots"}``.
    """
    removed = 0
    freed = 0
    roots = 0
    if store.dir.is_dir():
        for workspace_dir in store.dir.iterdir():
            if not workspace_dir.is_dir() or workspace_dir == store.tmp_dir:
                continue
            for root_dir in workspace_dir.iterdir():
                if not root_dir.is_dir() or not root_dir.name.isdigit():
                    continue
                roots += 1
                live = live_blobs(session, int(root_dir.name))
                for blob in root_dir.rglob("*"):
                    if not blob.is_file() or blob.name in live:
                        continue
                    size = blob.stat().st_size
                    if not dry_run:
                        try:
                            blob.unlink()
                        except OSError:
                            logger.warning("could not remove blob %s", blob, exc_info=True)
                            continue
                    removed += 1
                    freed += size
    tmp_removed = 0 if dry_run else store.sweep_tmp()
    counts = {
        "blobs_removed": removed,
        "bytes_freed": freed,
        "tmp_removed": tmp_removed,
        "roots": roots,
    }
    logger.info("blob store gc: %s", counts)
    return counts
