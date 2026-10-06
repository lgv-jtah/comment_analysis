"""Durable, single-writer, whole-comment benchmark checkpoints."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid
from urllib.parse import urlsplit

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from whole_process import OUTPUT_COLUMNS_V2

DEFAULT_EXECUTION_SETTINGS = {
    "max_retries": 3, "retry_base_delay": 1.0, "retry_max_delay": 30.0,
    "timeout": {"connect": 10.0, "read": 600.0, "write": 30.0, "pool": 10.0},
    "rate_limit_delay": 0.5,
}


BUDGET_KEYS = ("max_wall_seconds", "max_total_tokens", "max_http_attempts")
EXECUTION_KEYS = {"deadline", "http_attempts", "total_tokens", "unknown_usage_responses", "stop_reason"}
STOP_REASONS = ("wall_clock", "total_tokens", "http_attempts")


class CheckpointFailure(BaseException):
    """Persistence must escape provider handlers that catch Exception."""


class BudgetExhausted(BaseException):
    """A persisted run budget is used up.

    BaseException, like CheckpointFailure, so LLMHandler's broad Exception
    handlers cannot turn a stop into a failed call and continue.
    """

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


def validate_budget(budget):
    """Normalize run limits: each a positive int or None, at least one set.

    max_wall_seconds becomes an absolute deadline when the run starts and is
    never reset on resume. max_http_attempts counts completion requests sent
    through the run's client, including retries (not the route-metadata
    check). max_total_tokens counts client-visible usage only; a request in
    flight when it is reached can overshoot it, so it is a soft limit.
    """
    if type(budget) is not dict or set(budget) != set(BUDGET_KEYS):
        raise ValueError(f"Budget must define exactly {BUDGET_KEYS}")
    for key, value in budget.items():
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"Budget {key} must be a positive integer or None")
    if all(value is None for value in budget.values()):
        raise ValueError("Budget must set at least one limit")
    return dict(budget)


class RowValidationError(ValueError):
    """Invalid canonical rows, with a fixed safe operational reason code."""

    code = "row_validation_failed"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def fingerprint(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def strict_load(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("Non-finite JSON number")

    value = json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)
    canonical(value)  # Also rejects overflowing JSON floats such as 1e999.
    return value


def checked_endpoint(value):
    from benchmark import sanitized_endpoint
    if value is None:
        return None
    url = urlsplit(value)
    if (url.username is not None or url.password is not None or "?" in value or "#" in value
            or sanitized_endpoint(value) is None):
        raise ValueError("Endpoint must not contain userinfo, query or fragment")
    return sanitized_endpoint(value)


def atomic_write(path, writer):
    """Same-directory replace with durable data and directory entry."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    data = canonical(value)
    atomic_write(path, lambda handle: handle.write(data))


class BenchmarkCheckpoint:
    def __init__(self, *, output_path, resume=False, recover_stale_lock=False, **preparation):
        if recover_stale_lock and not resume:
            raise ValueError("Stale recovery requires --resume")
        self.output = Path(output_path)
        if self.output.suffix != ".parquet":
            raise ValueError("Durable benchmark output must have a .parquet suffix")
        self.path = self.output.with_suffix(".checkpoint.json")
        self.partial = self.output.with_suffix(".partial.parquet")
        self.lock_path = self.output.with_suffix(".lock")
        self.preparation = preparation
        self.resume = resume
        self.recover = recover_stale_lock
        self.lock = None
        self.poisoned = False
        self.active = False

    def __enter__(self):
        from benchmark import prepare_benchmark, load_frozen_comments
        from context_builder import build_comment_context
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.lock_path.open("a+b")
        try:
            try:
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("Benchmark is locked by a live process") from None
            state = None
            if self.resume:
                if not self.path.exists():
                    raise ValueError("No durable checkpoint; historical runs cannot be resumed")
                envelope = strict_load(self.path)
                if (type(envelope) is not dict or set(envelope) != {"version", "checksum", "payload"}
                        or type(envelope["version"]) is not int or envelope["version"] != 1
                        or envelope["checksum"] != fingerprint(envelope["payload"])):
                    raise ValueError("Invalid checkpoint version or checksum")
                state = envelope["payload"]
                if type(state) is not dict or type(state.get("manifest")) is not dict:
                    raise ValueError("Invalid checkpoint payload")
                run_id = state["manifest"].get("run_id")
                if type(run_id) is not str or not run_id.startswith("benchmark_"):
                    raise ValueError("Invalid original run identity")
            else:
                run_id = None
                for suffix in (".parquet", ".manifest.json", ".expected.parquet", ".execution.json",
                               ".checkpoint.json", ".partial.parquet", ".telemetry.json"):
                    if self.output.with_suffix(suffix).exists():
                        raise FileExistsError("Use a fresh benchmark output basename")
                if self.output.exists():
                    raise FileExistsError("Output already exists")
            self.expected, self.manifest = prepare_benchmark(
                **self.preparation, output_path=self.output, persist=False, durable=True, run_id=run_id)
            self.budget = self.manifest.get("budget")
            self.identities = []
            for entry in load_frozen_comments(self.preparation["source"])[:self.preparation.get("limit")]:
                ctx = build_comment_context(run_id=self.manifest["run_id"], **entry)
                self.identities.append([{
                    "contribution_id": ctx.contribution_id, "contributionContent": entry["contribution_text"],
                    "comment_text": ctx.comment_text, "key_message": km.text, "key_message_id": km.key_message_id,
                    "pair_id": pair, "run_id": ctx.run_id, "policy_version": "v2",
                    "stance_model_name": self.manifest["model_name"], "extraction_model_name": None,
                } for km, pair in zip(ctx.key_messages,
                    self.manifest["ordered_comments"][len(self.identities)]["pair_ids"], strict=True)])
            if state is not None:
                if canonical(state["manifest"]) != canonical(self.manifest):
                    raise ValueError("Resume contract changed")
                self.state = state
                self.validate()
                if state["session"] is not None and not self.recover:
                    raise ValueError("Stale session requires --resume --recover-stale-lock")
                if strict_load(self.output.with_suffix(".manifest.json")) != self.manifest:
                    raise ValueError("Manifest artifact mismatch")
                cohort = pd.read_parquet(self.output.with_suffix(".expected.parquet"))
                if canonical(cohort.to_dict("records")) != canonical(self.manifest["expected_pairs"]):
                    raise ValueError("Expected cohort artifact mismatch")
                self._validate_export(self.partial, partial=True)
                self._validate_export(self.output, partial=False)
            else:
                self.state = {"manifest": self.manifest, "manifest_fingerprint": fingerprint(self.manifest),
                    "records": [], "completed_comments": 0, "completed_pairs": 0,
                    "published": False, "session": None, "sessions": [], "execution": {}}
                if self.budget is not None:
                    seconds = self.budget["max_wall_seconds"]
                    self.state["execution"] = {
                        "deadline": time.time() + seconds if seconds is not None else None,
                        "http_attempts": 0, "total_tokens": 0, "unknown_usage_responses": 0,
                        "stop_reason": None,
                    }
                self.validate()
                atomic_json(self.output.with_suffix(".manifest.json"), self.manifest)
                atomic_write(self.output.with_suffix(".expected.parquet"),
                             lambda handle: self.expected.to_parquet(handle, index=False))
            self.state["session"] = uuid.uuid4().hex
            self.save()
            self.active = True
            return self
        except BaseException:
            self._unlock()
            raise

    def _unlock(self):
        if self.lock is not None:
            self.lock.close()
            self.lock = None

    def __exit__(self, kind, error, traceback):
        try:
            if not self.poisoned:
                if kind is not None and not self.complete:
                    self.export_partial()
                self.state["session"] = None
                self.save()
        finally:
            self.active = False
            self._unlock()

    @property
    def completed_comments(self):
        return self.state["completed_comments"]

    @property
    def complete(self):
        return self.completed_comments == len(self.identities)

    def rows(self):
        return [row for record in self.state["records"] for row in record["rows"]]

    def frame(self):
        return pd.DataFrame(self.rows(), columns=OUTPUT_COLUMNS_V2)

    def validate_rows(self, rows, index):
        identities = self.identities[index]
        if type(rows) is not list or len(rows) != len(identities):
            raise RowValidationError("Incomplete comment target coverage")
        for row, identity in zip(rows, identities, strict=True):
            if type(row) is not dict or set(row) != set(OUTPUT_COLUMNS_V2):
                raise RowValidationError("Invalid canonical output columns")
            if any(canonical(row[k]) != canonical(v) for k, v in identity.items()):
                raise RowValidationError("Mixed row identity")
            for key in ("comment_evidence", "source_evidence", "review_reasons"):
                if type(row[key]) is not list or any(type(x) is not str for x in row[key]):
                    raise RowValidationError("Invalid row list type")
            if type(row["explanation"]) is not str:
                raise RowValidationError("Invalid explanation type")
            if row["status"] not in ("accepted", "needs_review", "failed"):
                raise RowValidationError("Invalid row status")
            for key in ("proposed_stance", "final_stance"):
                if row[key] not in (None, "in_favor", "against", "neutral"):
                    raise RowValidationError("Invalid stance")
            if row["basis"] is not None and type(row["basis"]) is not str:
                raise RowValidationError("Invalid basis")
            accepted_bases = ("general_approval", "explicit_target", "explicit_exclusion", "unaddressed", "no_position")
            if row["basis"] not in (None, "unresolved", *accepted_bases):
                raise RowValidationError("Unknown output basis")
            if row["status"] == "accepted":
                if (row["final_stance"] is None or row["final_stance"] != row["proposed_stance"]
                        or row["basis"] not in accepted_bases or row["review_reasons"]):
                    raise RowValidationError("Invalid accepted row")
            elif row["final_stance"] is not None:
                raise RowValidationError("Invalid non-accepted row")
            if row["status"] == "failed" and (row["proposed_stance"] is not None or row["basis"] is not None):
                raise RowValidationError("Invalid failed row")

    def validate(self):
        s = self.state
        if set(s) != {"manifest", "manifest_fingerprint", "records", "completed_comments", "completed_pairs",
                      "published", "session", "sessions", "execution"}:
            raise ValueError("Invalid checkpoint fields")
        if s["manifest_fingerprint"] != fingerprint(self.manifest):
            raise ValueError("Manifest fingerprint mismatch")
        if (type(s["records"]) is not list or type(s["completed_comments"]) is not int
                or type(s["completed_pairs"]) is not int or type(s["published"]) is not bool
                or type(s["sessions"]) is not list or type(s["execution"]) is not dict
                or s["session"] is not None and (type(s["session"]) is not str or not s["session"])):
            raise ValueError("Invalid checkpoint field types")
        if s["completed_comments"] != len(s["records"]) or not 0 <= len(s["records"]) <= len(self.identities):
            raise ValueError("Invalid completed prefix")
        for index, record in enumerate(s["records"]):
            if (type(record) is not dict or set(record) != {"ordinal", "context_sha256", "rows"}
                    or type(record["ordinal"]) is not int or record["ordinal"] != index
                    or record["context_sha256"] != self.manifest["ordered_comments"][index]["context_sha256"]):
                raise ValueError("Invalid comment prefix identity")
            self.validate_rows(record["rows"], index)
        if s["completed_pairs"] != len(self.rows()) or s["published"] and not self.complete:
            raise ValueError("Invalid completed counts or publication")
        routes = []
        for session in s["sessions"]:
            if (type(session) is not dict or set(session) != {"run_id", "endpoint", "verification_requested",
                    "provider_verified", "verification_status", "route"}
                    or session["run_id"] != self.manifest["run_id"]
                    or session["endpoint"] != self.manifest["endpoint"]
                    or type(session["provider_verified"]) is not bool
                    or session["verification_requested"] is not self.manifest["verification_requested"]
                    or session["verification_status"] not in ("verified", "failed", "not_requested")):
                raise ValueError("Invalid inference session")
            if session["provider_verified"]:
                from ionos_config import _is_ionos_routed
                route = session["route"]
                if (session["verification_requested"] is not True or session["verification_status"] != "verified" or type(route) is not dict
                        or set(route) != {"upstream_model", "api_base", "is_ionos"}
                        or type(route["upstream_model"]) is not str or route["is_ionos"] is not True
                        or not _is_ionos_routed(route["api_base"])
                        or checked_endpoint(route["api_base"]) != route["api_base"]):
                    raise ValueError("Invalid verified route")
                routes.append(route)
            elif session["route"] is not None or session["verification_status"] == "verified":
                raise ValueError("Invalid unverified route")
        if any(route != routes[0] for route in routes):
            raise ValueError("Provider route changed across sessions")
        execution = s["execution"]
        if self.manifest.get("budget") is None:
            if execution:
                raise ValueError("Execution counters without a budget")
        elif (set(execution) != EXECUTION_KEYS
                or not (execution["deadline"] is None or type(execution["deadline"]) is float)
                or (execution["deadline"] is None) != (self.manifest["budget"]["max_wall_seconds"] is None)
                or any(type(execution[k]) is not int or execution[k] < 0
                       for k in ("http_attempts", "total_tokens", "unknown_usage_responses"))
                or execution["stop_reason"] not in (None, *STOP_REASONS)):
            raise ValueError("Invalid execution counters")

    def save(self):
        if self.poisoned or self.lock is None:
            raise CheckpointFailure("Checkpoint is not writable")
        try:
            self.validate()
            atomic_json(self.path, {"version": 1, "checksum": fingerprint(self.state), "payload": self.state})
        except BaseException as error:
            self.poisoned = True
            raise CheckpointFailure("Checkpoint persistence failed") from error

    def commit_comment(self, rows):
        if not self.active or self.complete:
            raise ValueError("No active incomplete checkpoint")
        index = self.completed_comments
        self.validate_rows(rows, index)
        self.state["records"].append({"ordinal": index,
            "context_sha256": self.manifest["ordered_comments"][index]["context_sha256"], "rows": copy.deepcopy(rows)})
        self.state["completed_comments"] += 1
        self.state["completed_pairs"] += len(rows)
        self.save()

    def check_budget(self):
        """Raise BudgetExhausted, persisting the reason, if any limit is used up."""
        if self.budget is None:
            return
        execution = self.state["execution"]
        reason = execution["stop_reason"]
        if reason is None:
            if execution["deadline"] is not None and time.time() >= execution["deadline"]:
                reason = "wall_clock"
            elif (self.budget["max_total_tokens"] is not None
                    and execution["total_tokens"] >= self.budget["max_total_tokens"]):
                reason = "total_tokens"
            elif (self.budget["max_http_attempts"] is not None
                    and execution["http_attempts"] >= self.budget["max_http_attempts"]):
                reason = "http_attempts"
        if reason is not None:
            if execution["stop_reason"] is None:
                execution["stop_reason"] = reason
                self.save()
            raise BudgetExhausted(reason)

    def reserve_http_attempt(self):
        """Durably count an attempt before it is sent; refuse it when the budget is used up."""
        self.check_budget()
        self.state["execution"]["http_attempts"] += 1
        self.save()

    def record_usage(self, total_tokens):
        execution = self.state["execution"]
        if type(total_tokens) is int and total_tokens >= 0:
            execution["total_tokens"] += total_tokens
        else:
            execution["unknown_usage_responses"] += 1
        self.save()

    def client_hooks(self):
        """httpx event hooks that enforce the budget on every request of the run client."""
        async def on_request(request):
            self.reserve_http_attempt()

        async def on_response(response):
            total = None
            try:
                await response.aread()
                usage = response.json().get("usage") or {}
                total = usage.get("total_tokens")
            except Exception:
                pass
            self.record_usage(total)

        return {"request": [on_request], "response": [on_response]}

    def record_session(self, execution):
        self.state["sessions"].append(execution)
        self.save()

    def validate_invocation(self, **kwargs):
        from benchmark import prepare_benchmark
        if not self.active or self.poisoned:
            raise ValueError("Checkpoint lifecycle is not active")
        preparation = dict(self.preparation)
        run_id = kwargs.pop("run_id")
        preparation.update(kwargs)
        _, actual = prepare_benchmark(**preparation, output_path=self.output, persist=False,
                                     durable=True, run_id=run_id)
        if canonical(actual) != canonical(self.manifest):
            raise ValueError("Inference invocation differs from checkpoint contract")
        self.validate()

    def _validate_export(self, path, *, partial):
        if not path.exists():
            return
        table = pq.read_table(path)
        metadata = table.schema.metadata or {}
        if metadata.get(b"benchmark_owner") != fingerprint(self.manifest).encode():
            raise ValueError("Unrelated existing prediction artifact")
        n = len(table)
        prefix_lengths = {0}
        count = 0
        for record in self.state["records"]:
            count += len(record["rows"])
            prefix_lengths.add(count)
        if (n not in prefix_lengths or not partial and (not self.complete or n != len(self.rows()))
                or table.column_names != OUTPUT_COLUMNS_V2
                or canonical(table.to_pylist()) != canonical(self.rows()[:n])):
            raise ValueError("Prediction artifact disagrees with checkpoint")

    def _export(self, path):
        if not self.active or self.poisoned or self.lock is None:
            raise CheckpointFailure("Export requires a healthy active checkpoint lock")
        table = pa.Table.from_pandas(self.frame(), preserve_index=False)
        table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                               b"benchmark_owner": fingerprint(self.manifest).encode()})
        atomic_write(path, lambda handle: pq.write_table(table, handle))

    def export_partial(self):
        self._validate_export(self.partial, partial=True)
        self._export(self.partial)
        return self.frame()

    def publish(self):
        if not self.active or self.poisoned or self.lock is None:
            raise CheckpointFailure("Publication requires a healthy active checkpoint lock")
        if not self.complete:
            raise ValueError("Final publication requires complete coverage")
        self._validate_export(self.output, partial=False)
        if not self.output.exists():
            self._export(self.output)
        self.state["published"] = True
        self.save()
        return self.frame()
