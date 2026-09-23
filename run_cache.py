"""Content-addressed cache for LLM stage calls, enabling resumable runs.

STANCE_PIPELINE_IMPLEMENTATION_BRIEF.md section 8: "Add resumable runs and a
cache keyed by the complete input context, stage, model/settings,
prompt/schema/policy version, and demonstrations. Do not reuse cached
outputs when any of those change. [...] Pin and log decoding settings,
model ID, provider-reported version when available, prompt hash, schema
version, policy version, code commit, and input hashes."

The cache key folds in every one of those fields (except code_commit, kept
only for logging -- see CacheKeyInputs.cache_key), so a change to any of
them (model swapped, prompt edited, policy version bumped, decoding
settings changed) produces a different key and is never served a stale
cached answer. It's an ordinary invalidation-by-construction cache, not a
TTL or manual-bust cache.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def hash_text(text: str) -> str:
    """Stable hash for prompt/demonstration text, used to build CacheKeyInputs."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CacheKeyInputs:
    """Every field that determines whether a cached result is still valid.

    Attributes:
        context_hash: Hash/identity of the full input context (e.g. a
            dataset_builder.compute_pair_id, or a hash of a CommentContext's
            relevant fields).
        stage: Which LLM stage this call is for (e.g. "scope_attribution",
            "per_target_stance").
        model_name: The model identifier actually used.
        base_url: The endpoint base URL actually used.
        decoding_settings: Decoding parameters that affect output
            (temperature, max_tokens, ...).
        prompt_hash: Hash of the exact system+user prompt templates used.
        schema_version: The response schema's version string.
        policy_version: The policy version ("v2", "legacy", ...).
        demonstrations_hash: Hash of any few-shot demonstrations used, or
            None when there are none.
        code_commit: The git commit the code ran at, when known. Carried for
            provenance/logging only -- see cache_key()'s docstring for why
            it is deliberately excluded from the key itself.
    """

    context_hash: str
    stage: str
    model_name: str
    base_url: str
    decoding_settings: dict
    prompt_hash: str
    schema_version: str
    policy_version: str
    demonstrations_hash: str | None = None
    code_commit: str | None = None

    def cache_key(self) -> str:
        """Deterministic content-addressed key for this exact combination.

        code_commit is intentionally excluded: two runs of byte-identical
        prompt/schema/model/policy-version inputs at different commits
        (e.g. because an unrelated file changed) should still hit the same
        cache entry -- what invalidates a cache entry is a change to
        something that could change the *output*, not the repository's
        commit graph.
        """
        payload = _canonical_json(
            {
                "context_hash": self.context_hash,
                "stage": self.stage,
                "model_name": self.model_name,
                "base_url": self.base_url,
                "decoding_settings": self.decoding_settings,
                "prompt_hash": self.prompt_hash,
                "schema_version": self.schema_version,
                "policy_version": self.policy_version,
                "demonstrations_hash": self.demonstrations_hash,
            }
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RunCache:
    """File-based content-addressed cache: one JSON file per cache key.

    Safe for resumable runs: a crashed/restarted run recomputes its
    CacheKeyInputs for each pending pair, and any pair whose exact
    combination was already computed is served from disk instead of
    re-calling the LLM. Pair identity is preserved because the key is a
    pure function of that combination, not of write order -- concurrent or
    out-of-order writes for different keys never collide.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        # Two-level fan-out (like git's object store) so the directory
        # doesn't accumulate an unwieldy number of files at one level for a
        # large run.
        return self._cache_dir / key[:2] / f"{key}.json"

    def get(self, key_inputs: CacheKeyInputs) -> dict | None:
        """Return the cached result for this exact key, or None on a miss."""
        path = self._path_for(key_inputs.cache_key())
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)
        return record.get("result")

    def set(self, key_inputs: CacheKeyInputs, result: dict) -> None:
        """Store a result for this exact key, along with its provenance."""
        path = self._path_for(key_inputs.cache_key())
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key_inputs": {
                "context_hash": key_inputs.context_hash,
                "stage": key_inputs.stage,
                "model_name": key_inputs.model_name,
                "base_url": key_inputs.base_url,
                "decoding_settings": key_inputs.decoding_settings,
                "prompt_hash": key_inputs.prompt_hash,
                "schema_version": key_inputs.schema_version,
                "policy_version": key_inputs.policy_version,
                "demonstrations_hash": key_inputs.demonstrations_hash,
                "code_commit": key_inputs.code_commit,
            },
            "result": result,
        }
        # Write to a temp file then rename: atomic on POSIX, so a crash
        # mid-write never leaves a corrupt cache entry a resumed run would
        # fail to parse.
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(record, f)
        tmp_path.replace(path)

    def __contains__(self, key_inputs: CacheKeyInputs) -> bool:
        return self._path_for(key_inputs.cache_key()).exists()
