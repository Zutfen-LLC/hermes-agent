"""Request-scoped provider runtime credentials for the API server.

Contract (advertised as ``features.provider_runtime_credentials``):

* ``X-Hermes-Provider-API-Key`` request header carries the SECRET provider API
  key for exactly one invocation. It is never accepted in the JSON body (a
  body-carried key is rejected 400) because clients durably retain request
  bodies — ``/v1/runs`` idempotency replay depends on that — and Hermes must
  not become the surface that makes body persistence a credential leak.
* ``provider_base_url`` body field carries the NON-secret base URL override.
* An explicit ``provider`` (body ``provider``/``provider_id``) is mandatory:
  a credential without an unambiguous provider identity is rejected.
* The override is request-scoped: no config/env/session/route mutation, no
  plaintext persistence anywhere (idempotency stores keep only a keyed HMAC
  fingerprint), and the next request without the header resolves normally.

Authorization: the header is honored only behind the normal API-server
authentication boundary (a configured, verified ``API_SERVER_KEY``). Room-grant
bearers and keyless test listeners are refused fail-closed.

Transport: the credential crosses the gateway connection verbatim; remote use
requires HTTPS/TLS or an equivalently protected trusted transport (Ops
Supervisor separately enforces HTTPS for non-loopback gateways).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import re
from urllib.parse import urlsplit
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

# Secret transport: purpose-specific header, never the durable JSON body.
PROVIDER_API_KEY_HEADER = "X-Hermes-Provider-API-Key"
# Non-secret routing override carried with the ordinary body fields.
PROVIDER_BASE_URL_FIELD = "provider_base_url"
# Body fields that would carry secret material — rejected, never read.
_BODY_SECRET_FIELDS = ("provider_api_key", "provider_credentials")

_MAX_API_KEY_LEN = 4096
_MAX_BASE_URL_LEN = 2048
_FORBIDDEN_SECRET_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_FINGERPRINT_DOMAIN = b"hermes-provider-credential-fp-v1"


class ProviderCredentialError(Exception):
    """Reject a credential-contract request with a stable, non-secret diagnostic."""

    def __init__(self, message: str, *, status: int = 400, code: str = "invalid_provider_credential"):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(slots=True)
class ProviderCredentialOverride:
    """One invocation's caller-supplied provider runtime (plaintext lives in
    memory only, for the lifetime of the request/run task)."""

    api_key: Optional[str] = None
    base_url: Optional[str] = None
    provider: str = ""
    # Keyed, non-reversible digest bound to the authenticated principal scope;
    # this is the ONLY derived value allowed to outlive the request.
    fingerprint: str = ""
    redaction_lease: Any = None

    def __repr__(self) -> str:  # defensive: keeps any repr that reaches a log safe
        return (f"<ProviderCredentialOverride provider={self.provider!r} "
                f"api_key={'set' if self.api_key else 'unset'} "
                f"base_url={self.base_url!r} fingerprint={self.fingerprint!r}>")


def _clean_secret(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _clean_base_url(value: Any) -> Optional[str]:
    if not isinstance(value, str) or not value or len(value) > _MAX_BASE_URL_LEN:
        return None
    if any(char.isspace() or ord(char) < 0x20 or ord(char) == 0x7f for char in value) or "\\" in value:
        return None
    if not value.startswith(("http://", "https://")):
        return None
    try:
        parsed = urlsplit(value)
        host = parsed.hostname
        if (parsed.scheme not in ("http", "https") or not host
                or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or "?" in value or "#" in value
                or "%" in parsed.netloc or parsed.netloc.endswith(":")
                or re.fullmatch(r"0[xX][0-9a-fA-F]+", host)):
            return None
        if parsed.port is not None and not (1 <= parsed.port <= 65535):
            return None
        if ":" in host:
            ipaddress.IPv6Address(host)
            if not parsed.netloc.startswith("["):
                return None
        elif re.fullmatch(r"[0-9.]+", host):
            ipaddress.IPv4Address(host)
        elif (len(host) > 253 or not all(
                0 < len(label) <= 63 and re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?", label)
                for label in host.split("."))):
            return None
    except ValueError:
        return None
    return value


def principal_fingerprint_secret(adapter: Any) -> bytes:
    """Load the installation's gateway-only signing secret for credential HMACs.

    No client-known bearer (including API_SERVER_KEY) may key durable provider
    fingerprints. An unavailable/corrupt/read-only installation secret rejects
    admission; never substitute another key and redefine existing replays.
    """
    from gateway.hosted_room_peer import gateway_room_grant_secret
    try:
        key = gateway_room_grant_secret()
    except Exception:
        raise ProviderCredentialError(
            "Gateway-only fingerprint secret is unavailable; credential request cannot be admitted.",
            status=503, code="provider_credential_fingerprint_unavailable") from None
    if not isinstance(key, bytes) or not key:
        raise ProviderCredentialError(
            "Gateway-only fingerprint secret is unavailable; credential request cannot be admitted.",
            status=503, code="provider_credential_fingerprint_unavailable")
    return key


def credential_fingerprint(adapter: Any, principal_scope: str, credential: ProviderCredentialOverride) -> str:
    """Bind the effective request-scoped provider runtime identity to replay checks.

    Principal scope, explicit provider, API-key identity, and provider base URL
    determine the runtime. A changed base URL selects a different runtime and
    must never replay the prior endpoint's result. After structural validation,
    the base URL is non-secret and is bound verbatim: normalizing near-identical
    spellings could equate distinct endpoints, while treating distinct spellings
    as distinct only recomputes, the fail-safe direction. URL-less credentials
    retain the pre-correction message so existing durable replays survive.

    The digest is a keyed HMAC, not a bare hash: an attacker holding the
    durable store must not be able to trial-dictionary the credential offline
    against an unsalted digest.
    """
    key = principal_fingerprint_secret(adapter)
    parts = (principal_scope, credential.provider or "", credential.api_key or "")
    if credential.base_url:
        parts += (credential.base_url,)
    message = "\0".join(parts).encode("utf-8")
    return hmac.new(key, _FINGERPRINT_DOMAIN + message, hashlib.sha256).hexdigest()


def _body_secret_error(secret_field: str) -> ProviderCredentialError:
    return ProviderCredentialError(
        f"Provider API keys must be supplied via the {PROVIDER_API_KEY_HEADER} header, "
        f"never the request body ('{secret_field}' would be durably retained by clients "
        "and idempotency replay).",
        code="provider_credential_in_body")


def extract_provider_credential(
    adapter: Any, request: Any, body: Dict[str, Any], *, scope_fn: Callable[[], str],
) -> Optional[ProviderCredentialOverride]:
    """Parse + validate the credential contract for one request.

    Returns ``None`` when the request does not use the contract at all
    (backward-compatible no-op). Raises :class:`ProviderCredentialError` with a
    non-secret message for every otherwise-invalid shape. ``scope_fn`` is
    evaluated only when a credential is actually present.
    """
    is_body = isinstance(body, dict)
    header_key = _clean_secret(request.headers.get(PROVIDER_API_KEY_HEADER))
    # Register at the first point the authenticated handler accepts the key,
    # before any body validation, route lookup, fingerprinting, or resolver can
    # raise. The handler-task callback covers *all* early-return/error paths.
    from agent.redact import register_provider_credential_redaction
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    lease = register_provider_credential_redaction(header_key) if task is not None else None
    if task is not None and lease is not None:
        task.add_done_callback(lambda _done: lease.release())
    body_base_url = _clean_base_url(body.get(PROVIDER_BASE_URL_FIELD)) if is_body else None
    invalid_base_url = is_body and body.get(PROVIDER_BASE_URL_FIELD) is not None and body_base_url is None
    using_contract = header_key is not None or (is_body and body.get(PROVIDER_BASE_URL_FIELD) is not None)
    for secret_field in _BODY_SECRET_FIELDS:
        if is_body and body.get(secret_field) is not None:
            raise _body_secret_error(secret_field)
    if invalid_base_url:
        raise ProviderCredentialError(
            f"'{PROVIDER_BASE_URL_FIELD}' must be a structurally valid http(s) URL of at most {_MAX_BASE_URL_LEN} characters.",
            code="invalid_provider_base_url")
    if not using_contract:
        return None
    if header_key is None:
        raise ProviderCredentialError(
            f"'{PROVIDER_BASE_URL_FIELD}' requires {PROVIDER_API_KEY_HEADER} for a credentialed provider.",
            code="provider_api_key_required")
    # Fail closed on any authorization shape that is not the normal
    # API_SERVER_KEY bearer boundary (room grants, keyless test listeners).
    room_token = getattr(adapter, "_room_grant_token", None)
    if room_token and room_token(request):
        raise ProviderCredentialError(
            "Request-scoped provider credentials require API-key authentication.",
            status=403, code="provider_credential_auth_required")
    if not adapter._expected_api_key():
        raise ProviderCredentialError(
            "Request-scoped provider credentials require a configured API_SERVER_KEY.",
            status=403, code="provider_credential_auth_required")
    if header_key is not None:
        if len(header_key) > _MAX_API_KEY_LEN or _FORBIDDEN_SECRET_CHARS.search(header_key):
            raise ProviderCredentialError(
                f"{PROVIDER_API_KEY_HEADER} must be 1-{_MAX_API_KEY_LEN} visible characters.",
                code="invalid_provider_credential")
    if len(str(body.get(PROVIDER_BASE_URL_FIELD)).strip()) > _MAX_BASE_URL_LEN:
        raise ProviderCredentialError(
            f"'{PROVIDER_BASE_URL_FIELD}' must be an http(s) URL of at most {_MAX_BASE_URL_LEN} characters.",
            code="invalid_provider_base_url")
    # Explicit provider identity is mandatory: never guess from the key.
    provider = ""
    if is_body:
        provider = (adapter._clean_runtime_id(body.get("provider") or body.get("provider_id"), max_len=80) or "")
    if not provider:
        raise ProviderCredentialError(
            "An explicit 'provider' is required when request-scoped provider credentials are supplied.",
            code="provider_required_for_credential")
    credential = ProviderCredentialOverride(
        api_key=header_key, base_url=body_base_url, provider=provider,
        redaction_lease=lease)
    try:
        credential.fingerprint = credential_fingerprint(adapter, scope_fn(), credential)
    except BaseException:
        if lease is not None:
            lease.release()
        raise
    return credential


def resolve_credential_runtime(
    credential: ProviderCredentialOverride, *, target_model: Optional[str],
) -> Dict[str, Any]:
    """Resolve the provider runtime for a credential request through the SAME
    explicit-credential rung the CLI uses (``resolve_runtime_provider``), so
    api_mode / endpoint normalization / provider validation are shared — never
    a parallel provider stack.

    Raises :class:`ProviderCredentialError` (fail-closed) when the provider
    cannot carry an explicit request-scoped key (OAuth/external-process rungs
    ignore ``explicit_api_key``; a silent drop would send someone else's
    credential — or none — to the caller's stated provider).
    """
    if not credential.api_key:
        raise ProviderCredentialError(
            f"Request-scoped provider runtime requires {PROVIDER_API_KEY_HEADER}.",
            code="provider_api_key_required")
    # The header carries API keys only. A provider whose registered metadata mandates OAuth or an external
    # process has no API-key rung — an explicit value there would ride as its OAuth bearer (openai-codex's
    # explicit rung forwards any string), i.e. an OAuth token smuggled through an API-key header.
    from agent.auth_authority import AUTH_EXTERNAL_PROCESS, AUTH_OAUTH, provider_auth_mechanism
    if provider_auth_mechanism(credential.provider) in (AUTH_OAUTH, AUTH_EXTERNAL_PROCESS):
        raise ProviderCredentialError(
            f"Provider '{credential.provider}' does not accept request-scoped API keys "
            "(its credentials resolve through a login or external process).",
            code="provider_credential_unsupported")
    from hermes_cli.runtime_provider import resolve_runtime_provider
    try:
        runtime = resolve_runtime_provider(
            requested=credential.provider,
            explicit_api_key=credential.api_key or None,
            explicit_base_url=credential.base_url or None,
            target_model=target_model or None)
    except Exception:
        # A provider may include arbitrary caller-key bytes in its exception.
        # Never relay the underlying text, even through a traceback chain.
        raise ProviderCredentialError(
            "Provider resolution failed for request-scoped credentials.",
            code="provider_resolution_failed") from None
    if not isinstance(runtime, dict):
        raise ProviderCredentialError(
            "Provider resolution failed for request-scoped credentials.",
            code="provider_resolution_failed")
    if credential.api_key and runtime.get("api_key") != credential.api_key:
        raise ProviderCredentialError(
            f"Provider '{credential.provider}' does not accept request-scoped API keys "
            "(its credentials resolve through a login or external process).",
            code="provider_credential_unsupported")
    return runtime


_CREDENTIAL_RUNTIME_KEYS = ("provider", "api_mode", "base_url", "api_key", "command", "args")


def apply_credential_runtime(runtime_kwargs: Dict[str, Any], runtime: Dict[str, Any]) -> None:
    """Merge the resolved explicit runtime over ``runtime_kwargs`` IN PLACE and
    remove static credential sources so the caller-supplied values cannot be
    silently overridden (route/catalog keys, rotating credential pools).

    Precedence (documented contract): for a credential request the request's
    provider/model are authoritative for THIS invocation, and the request-scoped
    key/base URL beat every static Hermes credential for it. Nothing is written
    back — the next request without the contract resolves normally.
    """
    # A static external-process runtime must not linger under the explicit one.
    runtime_kwargs.pop("command", None)
    runtime_kwargs.pop("args", None)
    for key in _CREDENTIAL_RUNTIME_KEYS:
        value = runtime.get(key)
        if value is None:
            continue
        runtime_kwargs[key] = list(value) if key == "args" and isinstance(value, (list, tuple)) else value
    if not runtime.get("api_key"):
        runtime_kwargs.pop("api_key", None)
    if not runtime.get("base_url"):
        runtime_kwargs.pop("base_url", None)
    # A rotating pool would replace the caller's key with a static one.
    runtime_kwargs["credential_pool"] = None
    # Caller-key redaction is owned by the authenticated request and its worker
    # leases, not the long-lived browser-vault redaction registry.


def error_response(exc: ProviderCredentialError):
    """OpenAI-style error response for a rejected credential request."""
    from gateway.platforms.api_server import _error_response
    return _error_response(exc.message, exc.status, err_type="invalid_request_error", code=exc.code)
