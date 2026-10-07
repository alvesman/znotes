"""Unit tests for the RRF sweep plumbing (task 32).

The sweep itself needs the real stack; here only the in-process settings
override, the ranking key and the settings copy are covered.
"""

from __future__ import annotations

from eval.sweep_rrf import patched_get_settings, rank_key, settings_with_fusion

import mynotes.config
import mynotes.retrieval.pipeline
from mynotes.config import get_settings


def test_settings_with_fusion_updates_only_fusion() -> None:
    base = get_settings()
    patched = settings_with_fusion(base, 40, (0.3, 0.7))
    assert patched.retrieval.fusion.rank_constant == 40
    assert patched.retrieval.fusion.hybrid_weights == [0.3, 0.7]
    assert base.retrieval.fusion.rank_constant == 20  # original (tuned) default untouched
    assert patched.models == base.models
    assert patched.opensearch == base.opensearch


def test_patched_get_settings_swaps_and_restores() -> None:
    base = get_settings()
    patched = settings_with_fusion(base, 80, (0.7, 0.3))
    seen = []
    with patched_get_settings(patched):
        seen.append(mynotes.config.get_settings().retrieval.fusion.rank_constant)
        seen.append(mynotes.retrieval.pipeline.get_settings().retrieval.fusion.rank_constant)
    assert seen == [80, 80]
    assert mynotes.config.get_settings() is base
    assert mynotes.retrieval.pipeline.get_settings is not None


def test_rank_key_prefers_pre_rerank_then_post() -> None:
    better_pre = {
        "pre_rerank": {"mrr": 0.8, "doc_hit@5": 0.9},
        "post_rerank": {"mrr": 0.5},
    }
    same_pre_better_post = {
        "pre_rerank": {"mrr": 0.8, "doc_hit@5": 0.9},
        "post_rerank": {"mrr": 0.7},
    }
    worse_pre = {
        "pre_rerank": {"mrr": 0.7, "doc_hit@5": 0.9},
        "post_rerank": {"mrr": 0.9},
    }
    missing = {"pre_rerank": {}, "post_rerank": {}}
    assert rank_key(same_pre_better_post) > rank_key(better_pre)
    assert rank_key(better_pre) > rank_key(worse_pre)
    assert rank_key(worse_pre) > rank_key(missing) > (0.0, 0.0, -1.0)
