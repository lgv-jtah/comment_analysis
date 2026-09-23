from run_cache import CacheKeyInputs, RunCache, hash_text


def _key(**overrides):
    defaults = dict(
        context_hash="ctx-1",
        stage="scope_attribution",
        model_name="mistral-nemo",
        base_url="http://localhost:4000",
        decoding_settings={"temperature": 0.1, "max_tokens": 20000},
        prompt_hash="prompt-abc",
        schema_version="scope_attribution_v1",
        policy_version="v2",
    )
    defaults.update(overrides)
    return CacheKeyInputs(**defaults)


# ---------------------------------------------------------------------------
# hash_text
# ---------------------------------------------------------------------------


def test_hash_text_is_deterministic():
    assert hash_text("some prompt text") == hash_text("some prompt text")


def test_hash_text_differs_for_different_text():
    assert hash_text("a") != hash_text("b")


# ---------------------------------------------------------------------------
# CacheKeyInputs.cache_key -- invalidation-by-construction
# ---------------------------------------------------------------------------


def test_cache_key_deterministic_for_identical_inputs():
    assert _key().cache_key() == _key().cache_key()


def test_cache_key_changes_when_context_changes():
    assert _key(context_hash="ctx-1").cache_key() != _key(context_hash="ctx-2").cache_key()


def test_cache_key_changes_when_stage_changes():
    assert _key(stage="scope_attribution").cache_key() != _key(stage="per_target_stance").cache_key()


def test_cache_key_changes_when_model_changes():
    assert _key(model_name="mistral-nemo").cache_key() != _key(model_name="llama-31-405b-fp8").cache_key()


def test_cache_key_changes_when_base_url_changes():
    a = _key(base_url="http://localhost:4000")
    b = _key(base_url="https://openai.inference.de-txl.ionos.com/v1")
    assert a.cache_key() != b.cache_key()


def test_cache_key_changes_when_decoding_settings_change():
    a = _key(decoding_settings={"temperature": 0.1, "max_tokens": 20000})
    b = _key(decoding_settings={"temperature": 0.5, "max_tokens": 20000})
    assert a.cache_key() != b.cache_key()


def test_cache_key_changes_when_prompt_hash_changes():
    assert _key(prompt_hash="a").cache_key() != _key(prompt_hash="b").cache_key()


def test_cache_key_changes_when_schema_version_changes():
    a = _key(schema_version="scope_attribution_v1")
    b = _key(schema_version="scope_attribution_v2")
    assert a.cache_key() != b.cache_key()


def test_cache_key_changes_when_policy_version_changes():
    assert _key(policy_version="v2").cache_key() != _key(policy_version="legacy").cache_key()


def test_cache_key_changes_when_demonstrations_hash_changes():
    a = _key(demonstrations_hash=None)
    b = _key(demonstrations_hash="demo-hash-1")
    assert a.cache_key() != b.cache_key()


def test_cache_key_ignores_code_commit():
    # Deliberate design decision: a commit that doesn't change any of the
    # other keyed fields must not invalidate the cache -- see cache_key()'s
    # docstring.
    a = _key(code_commit="aaa111")
    b = _key(code_commit="bbb222")
    assert a.cache_key() == b.cache_key()


# ---------------------------------------------------------------------------
# RunCache
# ---------------------------------------------------------------------------


def test_get_returns_none_on_miss(tmp_path):
    cache = RunCache(tmp_path / "cache")

    assert cache.get(_key()) is None


def test_set_then_get_round_trips(tmp_path):
    cache = RunCache(tmp_path / "cache")
    key = _key()
    result = {"scope_type": "whole_contribution_approval", "explanation": "x"}

    cache.set(key, result)

    assert cache.get(key) == result


def test_contains_reflects_presence(tmp_path):
    cache = RunCache(tmp_path / "cache")
    key = _key()

    assert key not in cache
    cache.set(key, {"ok": True})
    assert key in cache


def test_different_keys_do_not_collide(tmp_path):
    cache = RunCache(tmp_path / "cache")
    key_a = _key(context_hash="ctx-a")
    key_b = _key(context_hash="ctx-b")

    cache.set(key_a, {"result": "a"})
    cache.set(key_b, {"result": "b"})

    assert cache.get(key_a) == {"result": "a"}
    assert cache.get(key_b) == {"result": "b"}


def test_cache_dir_is_created_if_missing(tmp_path):
    cache_dir = tmp_path / "does" / "not" / "exist" / "yet"

    RunCache(cache_dir)

    assert cache_dir.exists()


def test_set_overwrites_existing_entry_for_the_same_key(tmp_path):
    cache = RunCache(tmp_path / "cache")
    key = _key()

    cache.set(key, {"result": "first"})
    cache.set(key, {"result": "second"})

    assert cache.get(key) == {"result": "second"}


def test_resumed_run_skips_already_cached_pairs(tmp_path):
    """Simulates the actual resumability use case: a run that already
    computed some pairs, then "restarts" and must not recompute them."""
    cache = RunCache(tmp_path / "cache")
    pair_ids = ["pair-1", "pair-2", "pair-3"]

    # First (interrupted) run: only pair-1 and pair-2 got computed.
    for pid in pair_ids[:2]:
        cache.set(_key(context_hash=pid), {"pair_id": pid, "result": "computed"})

    # Resumed run: check what's already cached before recomputing.
    recomputed = []
    for pid in pair_ids:
        key = _key(context_hash=pid)
        if cache.get(key) is None:
            recomputed.append(pid)
            cache.set(key, {"pair_id": pid, "result": "computed"})

    assert recomputed == ["pair-3"]
    assert all(cache.get(_key(context_hash=pid)) is not None for pid in pair_ids)
