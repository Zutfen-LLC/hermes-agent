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

import hashlib
import hmac
import re
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

    def __repr__(self) -> str:  # defensive: keeps any repr that reaches a log safe
        return (f"<ProviderCredentialOverride provider={self.provider!r} "
                f"api_key={'set' if self.api_key else 'unset'} "
                f"base_url={self.base_url!r} fingerprint={self.fingerprint!r}>")


def _clean_secret(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _clean_base_url(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text.rstrip("/") or None


def principal_fingerprint_secret(adapter: Any) -> bytes:
    """Keying material for credential fingerprints.

    Preference order, both gateway-only values no API client can read:
    1. the installation's RoomLink signing secret (stable across restarts, so
       replay fingerprints survive a gateway restart);
    2. an HMAC of the configured API-server key (same derivation domain the
       hosted-room grants use).
    """
    try:
        from gateway.hosted_room_peer import gateway_room_grant_secret
        return gateway_room_grant_secret()
    except Exception:
        pass
    from gateway.hosted_room_peer import derive_room_grant_secret
    return derive_room_grant_secret(adapter._expected_api_key())


def credential_fingerprint(adapter: Any, principal_scope: str, credential: ProviderCredentialOverride) -> str:
    """Non-reversible, key-bound digest of the secret for idempotency replay
    comparison. Unsalted hashing is deliberately avoided: an attacker holding
    the durable store must not be able to trial-dictionary the credential
    offline against a bare digest."""
    key = principal_fingerprint_secret(adapter)
    message = "\0".join((
        principal_scope,
        credential.provider or "",
        credential.api_key or "",
    )).encode("utf-8")
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
    body_base_url = _clean_base_url(body.get(PROVIDER_BASE_URL_FIELD)) if is_body else None
    using_contract = header_key is not None or body_base_url is not None
    for secret_field in _BODY_SECRET_FIELDS:
        if is_body and body.get(secret_field) is not None:
            raise _body_secret_error(secret_field)
    if not using_contract:
        return None
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
    if body_base_url is not None:
        if len(body_base_url) > _MAX_BASE_URL_LEN or not body_base_url.lower().startswith(("http://", "https://")):
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
        api_key=header_key, base_url=body_base_url, provider=provider)
    credential.fingerprint = credential_fingerprint(adapter, scope_fn(), credential)
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
    from hermes_cli.runtime_provider import resolve_runtime_provider, format_runtime_provider_error
    try:
        runtime = resolve_runtime_provider(
            requested=credential.provider,
            explicit_api_key=credential.api_key or None,
            explicit_base_url=credential.base_url or None,
            target_model=target_model or None)
    except Exception as exc:
        raise ProviderCredentialError(
            f"Provider resolution failed for request-scoped credentials: {format_runtime_provider_error(exc)}",
            code="provider_resolution_failed") from exc
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
    # Belt for error/dump paths: exact-substring scrub of the caller's key in
    # any redacted text (registry patterns only cover known key shapes).
    try:
        from agent.redact import register_vault_redaction_value
        register_vault_redaction_value(str(runtime.get("api_key") or ""))
    except Exception:
        pass


def error_response(exc: ProviderCredentialError):
    """OpenAI-style error response for a rejected credential request."""
    from gateway.platforms.api_server import _error_response
    return _error_response(exc.message, exc.status, err_type="invalid_request_error", code=exc.code)
