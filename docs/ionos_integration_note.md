# IONOS integration decision

`STANCE_PIPELINE_IMPLEMENTATION_BRIEF.md` section 8 requires: "All new model
inference must use IONOS AI Model Hub. Do not fall back to another
provider. The requirement is the managed Model Hub, not a separate
self-hosted deployment."

## Decision: keep the existing LiteLLM proxy, verify its routing at runtime

This project's stance/extraction inference (`STANCE_LLM_*` /
`EXTRACTION_LLM_*`) is configured to call a local LiteLLM proxy
(`http://localhost:4000`, a separate shared project at
`/home/julien/projects/LiteLLM/`, also used by other deployments on the same
Docker network) rather than IONOS's endpoint directly. Investigating that
proxy's own config (`/home/julien/projects/LiteLLM/config.yaml`) and
querying its live `/v1/model/info` endpoint confirmed that the `mistral-nemo`
model name — the one actually used for both stance detection and extraction
— is itself routed to IONOS's OpenAI-compatible endpoint,
`https://openai.inference.de-txl.ionos.com/v1/`, exactly matching the URL
the brief cites. So today, this pipeline's inference is already 100% IONOS
under the hood; it's just fronted by a proxy hop rather than a direct call.

Given that, the decision (confirmed with the project owner) is:

- **Keep the proxy.** It costs nothing to satisfy the brief's intent (all
  inference lands on IONOS), and it keeps the shared infra's benefits: one
  place to rotate the real IONOS credential, Prometheus metrics already
  wired up, and — usefully for the brief's own model-comparison step
  (section 8's Mistral Small 24B / Llama 3.3 70B / GPT-OSS 120B matrix) —
  a single place that already lists several IONOS-hosted models side by
  side (`mistral-nemo`, `llama-31-405b-fp8`, `gptoss-120b`, confirmed live).
- **Don't build a second, direct-IONOS adapter that bypasses it.** That
  would duplicate credential management for no behavioral gain, and would
  mean this project owning a dependency (the raw IONOS endpoint/credential)
  that the shared proxy project already owns and rotates.
- **Verify, don't just assume, that the routing holds.** `ionos_config.py`'s
  `verify_ionos_only()` queries the proxy's `/v1/model/info` at the start of
  every v2 inference run (`WholeProcess.run_v2()`) and aborts before any LLM
  call if the configured model isn't IONOS-routed — so if the proxy's
  routing config ever changes (a model gets repointed to a different
  backend, or the configured model name is a typo), this fails loudly and
  immediately rather than silently sending traffic to the wrong provider.

## What this does *not* cover

The key-message-quality judge feature (`JUDGE_LLM_*`, `key_message_judge.py`)
routes directly to Anthropic (`claude-sonnet-4-6`), not through IONOS at
all, and not through this decision. It's a separate, pre-existing feature
the brief doesn't mention, and it's explicitly out of scope for this
implementation (confirmed with the project owner) — `verify_ionos_only()` is
only wired into the v2 stance-detection path, not the judge.

## If this needs to change later

If a future requirement genuinely needs a direct, proxy-free IONOS call
(e.g. the shared LiteLLM deployment becomes unavailable, or its ownership
changes), `ionos_config.py` already has the one fact that decision would
need: the confirmed direct endpoint is
`https://openai.inference.de-txl.ionos.com/v1/`, and the model ID IONOS
expects is the `litellm_params.model` value with the `openai/` prefix
stripped (e.g. `mistralai/Mistral-Nemo-Instruct-2407`).
