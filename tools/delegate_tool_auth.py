"""Bind a delegated child to one authentication authority before it is constructed (ops-supervisor#216).

A child route is provider + endpoint + auth mechanism + credential source, resolved together. The credential string
the parent happens to hold is only accepted when a canonical authority stands behind it: a pool entry that carries
it, the provider's own OAuth login store, a runtime the provider system just resolved, or an operator-configured
key. An OAuth-only provider (per its registered metadata) with no canonical OAuth authority fails closed here with a
stable, secret-free code, so the child is never built on an opaque inherited string and the parent keeps working.

A provider may accept several mechanisms (Nous: OAuth login and an explicit inference key —
``agent.auth_authority.provider_accepted_mechanisms``). There the credential's auth type is the selecting authority's:
its pool entry, the auth type the resolving rung stamped on the runtime, or the parent's bound authority — never the
provider's native login. An inherited credential none of those identify is bound ``undetermined``: it stays on its
own route and admits no rotation.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from agent.auth_authority import (
    AUTH_API_KEY, AUTH_CLOUD_SDK, AUTH_EXTERNAL_PROCESS, AUTH_NONE, AUTH_OAUTH, AuthAuthority, endpoint_identity,
    provider_accepted_mechanisms, provider_auth_mechanism, route_identity,
)

logger = logging.getLogger("tools.delegate_tool")

# Where the child's candidate credential came from. Only KEY_RUNTIME and pool entries are canonical by
# construction; KEY_PARENT must be backed by the parent's authority and KEY_EXPLICIT is an operator literal.
KEY_PARENT = "inherited"
KEY_EXPLICIT = "explicit"
KEY_RUNTIME = "runtime"
# Hermes' marker for endpoints that take no credential (see hermes_cli.model_switch); never a secret.
NO_KEY_PLACEHOLDER = "no-key-required"
# Auth type of an inherited, unpooled credential on a multi-mechanism provider when nothing canonical says which it
# is. Not a pooled mechanism, so the authority admits no rotation.
AUTH_UNDETERMINED = "undetermined"

# Stable diagnostic codes (the exception text never carries credential material or upstream error text).
AUTH_UNAVAILABLE = "delegation_auth_unavailable"
OAUTH_RELOGIN_REQUIRED = "delegation_oauth_relogin_required"
OAUTH_REFRESH_CONFLICT = "delegation_oauth_refresh_conflict"
AUTH_TYPE_MISMATCH = "delegation_auth_type_mismatch"
ROUTE_MISMATCH = "delegation_auth_route_mismatch"
PROFILE_AUTH_MISMATCH = "delegation_profile_auth_mismatch"


class DelegationAuthError(ValueError):
    """A child route has no usable authentication authority. ``ValueError`` so ``_build_children`` refuses the
    spawn with a tool error while the parent keeps running."""

    def __init__(self, code: str, provider: Any, detail: str):
        self.code = code
        super().__init__(f"Subagent authentication unavailable [{code}] for provider '{provider or '-'}': {detail}")


def _pool_entries(pool: Any) -> List[Any]:
    """The pool's entries, or [] for pools without entry metadata (legacy adapters, test doubles)."""
    entries_fn = getattr(pool, "entries", None)
    entries = entries_fn() if callable(entries_fn) else None
    return entries if isinstance(entries, list) else []


def _live(entry: Any) -> bool:
    """A DEAD entry (revoked / refresh failed) is no authority; cooling-down entries still are (the lease decides)."""
    return getattr(entry, "last_status", None) != "dead"


def _entry_for_key(entries: List[Any], key: Any) -> Any:
    if not isinstance(key, str) or not key:
        return None
    return next((e for e in entries if _live(e) and getattr(e, "runtime_api_key", None) == key), None)


def _entry_by_id(entries: List[Any], entry_id: Any) -> Any:
    if not isinstance(entry_id, str) or not entry_id:
        return None
    return next((e for e in entries if _live(e) and getattr(e, "id", None) == entry_id), None)


def parent_auth_type(parent_agent: Any) -> Optional[str]:
    """Auth type of the credential the parent runs on: its bound pool entry's, its own bound authority's (a nested
    child), else what its provider mandates. None when that is not determinable (multi-mechanism providers)."""
    entry = _entry_by_id(_pool_entries(getattr(parent_agent, "_credential_pool", None)),
                         getattr(parent_agent, "_credential_pool_entry_id", None))
    if entry is not None:
        return getattr(entry, "auth_type", None)
    authority = getattr(parent_agent, "_auth_authority", None)
    if isinstance(authority, AuthAuthority) and authority.auth_type != AUTH_UNDETERMINED:
        return authority.auth_type
    return provider_auth_mechanism(getattr(parent_agent, "provider", None))


def _parent_may_hold_oauth(parent_agent: Any) -> bool:
    """Whether the parent's credential may be an OAuth token: proven OAuth, or undetermined on a provider that
    accepts OAuth (or a token format that means OAuth). Fail closed: unknown on an OAuth-capable provider is OAuth."""
    known = parent_auth_type(parent_agent)
    if known is not None:
        return known == AUTH_OAUTH
    provider = getattr(parent_agent, "provider", None)
    key = getattr(parent_agent, "api_key", None)
    return AUTH_OAUTH in (provider_accepted_mechanisms(provider) or ()) or (
        isinstance(key, str) and _unpooled_auth_type(provider, key) == AUTH_OAUTH)


def _unpooled_auth_type(provider: Any, key: str) -> str:
    """Auth type of a credential no pool entry carries (token formats with one unambiguous meaning only)."""
    from agent.credential_pool import _normalize_pool_auth_type
    return _normalize_pool_auth_type(str(provider or ""), key, None)


def _authority(provider, base_url, auth_type, source, entry_id=None, profile=None) -> AuthAuthority:
    return AuthAuthority.for_route(provider, base_url, auth_type, source, entry_id, profile)


def _oauth_error(exc: Exception, provider: Any) -> DelegationAuthError:
    """Map a canonical OAuth resolution failure to a stable code without relaying its text."""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and ("_auth_missing" in code or code.endswith("_auth_invalid_shape")):
        return DelegationAuthError(AUTH_UNAVAILABLE, provider, "no OAuth login is configured for this provider; "
                                   f"run `hermes auth add {provider} --type oauth`.")
    if code == "refresh_token_reused":
        return DelegationAuthError(OAUTH_REFRESH_CONFLICT, provider, "the OAuth refresh token was already used by "
                                   "another client; sign in again with `hermes auth add "
                                   f"{provider} --type oauth`.")
    if getattr(exc, "relogin_required", False):
        return DelegationAuthError(OAUTH_RELOGIN_REQUIRED, provider, "the OAuth login is expired or revoked; sign "
                                   f"in again with `hermes auth add {provider} --type oauth`.")
    return DelegationAuthError(AUTH_UNAVAILABLE, provider, f"no usable OAuth login ({code or type(exc).__name__}); "
                               f"run `hermes auth add {provider} --type oauth`.")


def _canonical_oauth(provider: str, base_url: Any, model: Any, entries: List[Any], profile: Optional[str]):
    """``(authority, credential)`` for an OAuth-only provider when no pool entry carries the child's credential.

    Pooled OAuth first — a live (not DEAD) OAuth entry for the child's endpoint, the same set the pool would rotate
    the parent across — then the provider's own login store, which refreshes and writes through canonically.
    Neither: fail closed."""
    probe = _authority(provider, base_url, AUTH_OAUTH, "-")
    pooled = next((e for e in entries if _live(e) and (getattr(e, "runtime_api_key", "") or "")
                   and probe.admits(e, base_url)), None)
    if pooled is not None:
        return _authority(provider, base_url, AUTH_OAUTH, getattr(pooled, "source", None), pooled.id, profile), \
            pooled.runtime_api_key
    from hermes_cli.runtime_provider import resolve_oauth_store_runtime
    try:
        runtime = resolve_oauth_store_runtime(provider, target_model=model if isinstance(model, str) else None)
    except Exception as exc:
        raise _oauth_error(exc, provider) from None
    key = (runtime or {}).get("api_key")
    if not isinstance(key, str) or not key:
        raise DelegationAuthError(AUTH_UNAVAILABLE, provider, "no OAuth login or pooled OAuth credential exists "
                                  f"for this provider; run `hermes auth add {provider} --type oauth`.")
    # Enforceable route identity (scheme included): an https login never authorizes an http route.
    if route_identity(runtime.get("base_url")) != route_identity(base_url):
        raise DelegationAuthError(ROUTE_MISMATCH, provider, f"the OAuth login serves "
                                  f"{endpoint_identity(runtime.get('base_url'))}, not the child's endpoint "
                                  f"{endpoint_identity(base_url)}.")
    return _authority(provider, base_url, AUTH_OAUTH, runtime.get("source") or "oauth-store", None, profile), key


def bind_child_authority(
    rt: dict, *, parent_agent: Any, pool: Any, key_origin: Optional[str], key_source: Optional[str] = None,
    key_auth_type: Optional[str] = None, same_route: bool, expected_auth_type: Optional[str] = None,
    profile: Optional[str] = None,
) -> AuthAuthority:
    """Resolve the child's authentication authority and bind ``rt["api_key"]`` to its credential (in place).

    *key_origin* says where ``rt["api_key"]`` came from (``KEY_PARENT`` / ``KEY_EXPLICIT`` / ``KEY_RUNTIME`` /
    None); *key_auth_type* the auth type the resolving rung stamped on a ``KEY_RUNTIME`` credential; *same_route*
    whether the child runs the parent's exact provider + endpoint. Raises
    :class:`DelegationAuthError` when no canonical authority exists."""
    provider, base_url, model = rt.get("provider"), rt.get("base_url"), rt.get("model")
    mechanism = provider_auth_mechanism(provider)
    key = rt.get("api_key") if isinstance(rt.get("api_key"), str) else None
    acp_command = rt.get("acp_command")
    if (isinstance(acp_command, str) and acp_command) or mechanism == AUTH_EXTERNAL_PROCESS:
        authority = _authority(provider, base_url, AUTH_EXTERNAL_PROCESS, "external-process", profile=profile)
    elif mechanism == AUTH_CLOUD_SDK:
        authority = _authority(provider, base_url, AUTH_CLOUD_SDK, key_source or "cloud-sdk", profile=profile)
    else:
        authority = _bind_credential(rt, parent_agent, pool, provider, base_url, model, mechanism, key, key_origin,
                                     key_source, key_auth_type, same_route, profile)
    if expected_auth_type and authority.auth_type != expected_auth_type:
        raise DelegationAuthError(PROFILE_AUTH_MISMATCH, provider, f"the profile requires auth_type="
                                  f"{expected_auth_type} but the route resolved auth_type={authority.auth_type}.")
    return authority


def _bind_credential(rt, parent_agent, pool, provider, base_url, model, mechanism, key, key_origin, key_source,
                     key_auth_type, same_route, profile) -> AuthAuthority:
    accepted = provider_accepted_mechanisms(provider)
    if key_origin == KEY_PARENT and not same_route and (mechanism == AUTH_OAUTH or _parent_may_hold_oauth(parent_agent)):
        # The parent's credential belongs to the parent's authority: an OAuth token never leaves its own route, and
        # an OAuth-only route never adopts whatever the parent holds.
        key, key_origin = None, None
        rt["api_key"] = None
    entries = _pool_entries(pool)
    entry = None
    if key_origin == KEY_PARENT and same_route:
        # The parent's bound entry by id survives an OAuth refresh the pool applied after the parent read its token.
        entry = _entry_by_id(entries, getattr(parent_agent, "_credential_pool_entry_id", None))
    entry = entry or _entry_for_key(entries, key)
    if entry is not None and accepted is not None and getattr(entry, "auth_type", None) not in accepted:
        if key_origin == KEY_EXPLICIT:
            raise DelegationAuthError(AUTH_TYPE_MISMATCH, provider, f"pool entry {entry.id} is "
                                      f"auth_type={getattr(entry, 'auth_type', None)}, but this provider only accepts "
                                      f"{', '.join(sorted(accepted))}.")
        # An inherited or auto-selected entry of a mechanism the provider does not accept is not its authority:
        # resolve the real one.
        logger.info("subagent auth: pool entry %s is auth_type=%s, which provider %s does not accept; resolving its "
                    "authority instead", entry.id, getattr(entry, "auth_type", None), provider)
        entry, key_origin = None, None
    if entry is not None:
        rt["api_key"] = entry.runtime_api_key
        return _authority(provider, base_url, getattr(entry, "auth_type", None) or AUTH_API_KEY,
                          getattr(entry, "source", None), entry.id, profile)
    if mechanism == AUTH_OAUTH:
        if key_origin == KEY_EXPLICIT:
            raise DelegationAuthError(AUTH_TYPE_MISMATCH, provider, "a configured delegation api_key cannot "
                                      "authenticate an OAuth-only provider; remove it to use the provider's login.")
        if key_origin == KEY_RUNTIME and key:
            # resolve_runtime_provider just produced it from the provider's own store (no pool entry carries it).
            return _authority(provider, base_url, AUTH_OAUTH, key_source or "oauth-store", None, profile)
        authority, rt["api_key"] = _canonical_oauth(provider, base_url, model, entries, profile)
        return authority
    if key and key != NO_KEY_PLACEHOLDER:
        source = key_source if key_origin == KEY_RUNTIME and key_source else (key_origin or "inherited")
        return _authority(provider, base_url, _unpooled_key_auth_type(parent_agent, provider, key, key_origin,
                                                                       key_auth_type, accepted), source, None, profile)
    # Local / no-auth endpoint: no credential is manufactured. The placeholder keeps AIAgent on the child's own
    # endpoint (a falsy key would re-route it through the configured provider's credentials instead).
    rt["api_key"] = NO_KEY_PLACEHOLDER if base_url else None
    return _authority(provider, base_url, AUTH_NONE, "none", None, profile)


def _unpooled_key_auth_type(parent_agent, provider, key, key_origin, key_auth_type, accepted) -> str:
    """Auth type of a credential no pool entry carries, from the authority that selected it: the rung's stamp for a
    runtime key, an operator literal is an API key (unless its format says OAuth), an inherited key is the parent's.
    A multi-mechanism provider with nothing canonical to go on is undetermined, never a guess."""
    if key_origin == KEY_RUNTIME and key_auth_type in (AUTH_OAUTH, AUTH_API_KEY):
        return key_auth_type
    if key_origin == KEY_PARENT:
        inherited = parent_auth_type(parent_agent)
        if inherited in (AUTH_OAUTH, AUTH_API_KEY):
            return inherited
    by_format = _unpooled_auth_type(provider, key)
    if key_origin == KEY_EXPLICIT or by_format == AUTH_OAUTH or accepted is None or len(accepted) == 1:
        return by_format
    return AUTH_UNDETERMINED


def log_child_auth_route(child: Any, authority: Optional[AuthAuthority]) -> None:
    """Non-secret route facts for the child's bound authentication (never a key, token or bearer value)."""
    logger.info("subagent auth route: model=%s %s", getattr(child, "model", None) or "-",
                authority.describe() if authority is not None else "authority=unbound")
