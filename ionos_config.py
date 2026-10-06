"""Verifies the configured LLM endpoint actually routes to IONOS AI Model Hub.

Per STANCE_PIPELINE_IMPLEMENTATION_BRIEF.md section 8: "All new model
inference must use IONOS AI Model Hub. Do not fall back to another
provider." This repo satisfies that today via the local LiteLLM proxy
(a separate shared project at /home/julien/projects/LiteLLM/), which
already routes the configured stance/extraction model
(mistral-nemo -> openai/mistralai/Mistral-Nemo-Instruct-2407 @
https://openai.inference.de-txl.ionos.com/v1/) to IONOS's OpenAI-compatible
endpoint -- confirmed live. See docs/ionos_integration_note.md for the
decision not to bypass the proxy with a direct IONOS adapter.

This module verifies that routing holds at run time rather than assuming it
stays true forever, using the proxy's LiteLLM-specific ``/v1/model/info``
endpoint. A plain OpenAI-style ``/v1/models`` call does not expose upstream
routing at all (confirmed empirically: every model there reports
``"owned_by": "openai"`` regardless of actual backend) -- ``/v1/model/info``
is the one that actually answers the question.
"""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

# New regional hosts require explicit approval rather than substring trust.
IONOS_APPROVED_HOSTS = {"openai.inference.de-txl.ionos.com"}

DEFAULT_TIMEOUT = 10.0


class IonosVerificationError(Exception):
    """Raised when a model does not resolve to an IONOS-routed entry."""


@dataclass(frozen=True)
class ModelRouteInfo:
    """What the proxy reports about where a configured model name actually routes.

    Attributes:
        model_name: The name callers configure (e.g. STANCE_LLM_MODEL_NAME).
        upstream_model: The litellm_params.model value (e.g.
            "openai/mistralai/Mistral-Nemo-Instruct-2407").
        api_base: The upstream endpoint this model is routed to.
        is_ionos: Whether api_base resolves to an IONOS host.
    """

    model_name: str
    upstream_model: str | None
    api_base: str | None
    is_ionos: bool


def _is_ionos_routed(api_base: str | None) -> bool:
    try:
        url = urlsplit(api_base or "")
        return (url.scheme == "https" and url.hostname in IONOS_APPROVED_HOSTS
                and url.port in (None, 443) and not url.username and not url.password
                and not url.query and not url.fragment)
    except ValueError:
        return False


def discover_models(
    base_url: str, api_key: str, timeout: float = DEFAULT_TIMEOUT
) -> dict[str, ModelRouteInfo]:
    """Query the proxy's /v1/model/info endpoint and classify each model's routing.

    Args:
        base_url: The LiteLLM proxy's base URL (e.g. http://localhost:4000).
        api_key: Bearer token for the proxy.
        timeout: Request timeout in seconds.

    Returns:
        A dict of model_name -> ModelRouteInfo for every model the proxy
        reports.

    Raises:
        httpx.HTTPStatusError: If the request itself fails (bad auth, proxy
            unreachable, ...) -- a configuration error surfaced directly
            rather than silently swallowed.
    """
    root = base_url.rstrip("/")
    url = f"{root if root.endswith('/v1') else root + '/v1'}/model/info"
    response = httpx.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    routes: dict[str, ModelRouteInfo] = {}
    for entry in payload.get("data", []):
        model_name = entry.get("model_name")
        if model_name is None:
            continue
        params = entry.get("litellm_params") or {}
        api_base = params.get("api_base")
        previous = routes.get(model_name)
        routes[model_name] = ModelRouteInfo(
            model_name=model_name,
            upstream_model=params.get("model"),
            api_base=api_base,
            is_ionos=_is_ionos_routed(api_base) and (previous is None or (
                previous.is_ionos and previous.upstream_model == params.get("model")
                and (previous.api_base or "").rstrip("/") == (api_base or "").rstrip("/"))),
        )
    return routes


def verify_ionos_only(
    model_name: str, base_url: str, api_key: str, timeout: float = DEFAULT_TIMEOUT
) -> ModelRouteInfo:
    """Fail fast unless the configured model resolves to an IONOS-routed entry.

    Intended to run once at the start of any inference run (CLI benchmark,
    WholeProcess.run_v2, ...) so a misconfiguration is caught immediately
    rather than after burning time/cost on calls to the wrong provider.

    Args:
        model_name: The model name configured for inference (e.g. the value
            of STANCE_LLM_MODEL_NAME).
        base_url: The LiteLLM proxy's base URL.
        api_key: Bearer token for the proxy.
        timeout: Request timeout in seconds.

    Returns:
        The matched ModelRouteInfo, confirmed IONOS-routed.

    Raises:
        IonosVerificationError: If the model is unknown to the proxy or is
            routed to a non-IONOS backend.
    """
    if _is_ionos_routed(base_url):
        return ModelRouteInfo(model_name, model_name, base_url, True)
    routes = discover_models(base_url, api_key, timeout=timeout)
    info = routes.get(model_name)
    if info is None:
        raise IonosVerificationError(
            f"Model {model_name!r} is not present in the proxy's model list "
            f"({base_url}) -- cannot verify it is IONOS-routed."
        )
    if not info.is_ionos:
        raise IonosVerificationError(
            f"Model {model_name!r} is routed to {info.api_base!r}, not IONOS. "
            "STANCE_PIPELINE_IMPLEMENTATION_BRIEF.md section 8 requires "
            "IONOS-only inference for this pipeline."
        )
    return info
