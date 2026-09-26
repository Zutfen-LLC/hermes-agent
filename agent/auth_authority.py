"""The authentication authority a session runs on: provider, endpoint, auth mechanism and credential source.

An ``AuthAuthority`` names WHICH credential authority backs a route without carrying the credential itself, so it
can travel into logs, errors and session metadata. Delegated children bind to one at construction and keep it for
their lifetime: credential rotation may move between entries of the same authority, never to another auth type or
endpoint (ops-supervisor#216). The endpoint is enforced as a canonical route (scheme, host, effective port, path):
``https`` and ``http`` spellings of one host are different authorities.

A provider's native login mechanism comes from its registered metadata (``PROVIDER_REGISTRY`` rows, which provider
plugins mirror their ``ProviderProfile.auth_type`` into), never from provider or model names. Native is not
exclusive: the mechanisms a provider ACCEPTS add ``api_key`` when its canonical explicit-credential rung takes one
(Nous: OAuth login and an explicit inference key). What a given credential IS comes from the authority that
selected it (its pool entry, the rung that resolved it), not from the provider's native mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

AUTH_OAUTH = "oauth"
AUTH_API_KEY = "api_key"
AUTH_EXTERNAL_PROCESS = "external_process"
AUTH_CLOUD_SDK = "cloud_sdk"
AUTH_NONE = "none"

# Registry ``auth_type`` values that fix one mechanism for the provider's endpoint. Every ``oauth*`` value
# (oauth_device_code, oauth_external, oauth_minimax, plugin-declared variants) means OAuth only. ``api_key`` rows
# are not a constraint: some (anthropic, openrouter, copilot) also hold pooled OAuth credentials.
_REGISTRY_MECHANISMS = {"external_process": AUTH_EXTERNAL_PROCESS, "aws_sdk": AUTH_CLOUD_SDK, "vertex": AUTH_CLOUD_SDK}


def _canonical_provider(provider: Any) -> str:
    """Canonical provider id; aliases (``codex`` → ``openai-codex``) resolve through the provider registry."""
    if not isinstance(provider, str) or not provider.strip():
        return ""
    from providers import get_provider_profile
    raw = provider.strip().lower()
    return getattr(get_provider_profile(raw), "name", None) or raw


def provider_native_mechanism(provider: Any) -> Optional[str]:
    """The login mechanism *provider*'s registered metadata makes native (OAuth, external process, cloud SDK), or
    None for ``api_key`` rows and unregistered providers. Native is not exclusive: see
    :func:`provider_accepted_mechanisms`."""
    canonical = _canonical_provider(provider)
    if not canonical:
        return None
    from hermes_cli.auth_plugin_providers import registry_lookup
    registry_auth_type = str(getattr(registry_lookup(canonical), "auth_type", "") or "")
    if registry_auth_type.startswith("oauth"):
        return AUTH_OAUTH
    return _REGISTRY_MECHANISMS.get(registry_auth_type)


def provider_accepted_mechanisms(provider: Any) -> Optional[frozenset]:
    """Every auth mechanism *provider*'s runtime accepts, or None when its metadata imposes no constraint (``api_key``
    rows and unregistered/custom providers).

    The native login mechanism plus ``api_key`` when the canonical explicit-credential rung takes an API key
    (``hermes_cli.runtime_provider.explicit_credential_auth_type``): Nous accepts its OAuth login and an explicit
    inference key; openai-codex's explicit rung carries an OAuth bearer, so Codex stays OAuth-only."""
    native = provider_native_mechanism(provider)
    if native is None:
        return None
    from hermes_cli.runtime_provider import explicit_credential_auth_type
    accepted = {native}
    if explicit_credential_auth_type(_canonical_provider(provider)) == AUTH_API_KEY:
        accepted.add(AUTH_API_KEY)
    return frozenset(accepted)


def provider_auth_mechanism(provider: Any) -> Optional[str]:
    """The one auth mechanism *provider* mandates, or None when it accepts several (or is unconstrained)."""
    accepted = provider_accepted_mechanisms(provider)
    return next(iter(accepted)) if accepted is not None and len(accepted) == 1 else None


def provider_accepts_api_key(provider: Any) -> bool:
    """Whether *provider* can authenticate with an API key (unconstrained providers can)."""
    accepted = provider_accepted_mechanisms(provider)
    return accepted is None or AUTH_API_KEY in accepted


def route_identity(base_url: Any) -> str:
    """Enforceable route identity: ``normalize_route_base_url`` (scheme, host, effective port, path, query), so only
    proven-equivalent spellings compare equal and ``http`` never equals ``https``. May carry URL userinfo — compare
    it, never log or persist it (:func:`endpoint_identity` is the display form)."""
    from hermes_cli.route_identity import normalize_route_base_url
    return normalize_route_base_url(base_url)


def endpoint_identity(base_url: Any) -> str:
    """Non-secret display identity of a route: scheme, host, effective port and path (userinfo and query never
    included). Diagnostics only; authority checks compare :func:`route_identity`."""
    from urllib.parse import urlsplit
    normalized = route_identity(base_url)
    try:
        parts = urlsplit(normalized)
        port = parts.port
    except ValueError:
        return "-"
    if not parts.hostname:
        return "-"
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    # normalize_route_base_url already dropped default ports; a port left over is significant.
    return f"{parts.scheme}://{host}{f':{port}' if port else ''}{parts.path.rstrip('/')}"


@dataclass(frozen=True)
class AuthAuthority:
    """Secret-free identity of the credential authority behind one route.

    ``auth_source`` is a category (``device_code``, ``env:GLM_API_KEY``, ``hermes-auth-store``, ``explicit``,
    ``inherited``, ...), never credential material; ``entry_id`` names the concrete pool entry when pooled.
    """

    provider: str
    endpoint: str
    auth_type: str
    auth_source: str
    entry_id: Optional[str] = None
    profile: Optional[str] = None
    # Enforceable route (:func:`route_identity`) frozen at bind time; ``endpoint`` is its display form. Excluded
    # from repr, describe() and metadata because URL userinfo may be credential material.
    route: str = field(default="", repr=False)

    @classmethod
    def for_route(cls, provider: Any, base_url: Any, auth_type: str, auth_source: Any, entry_id: Optional[str] = None,
                  profile: Optional[str] = None) -> "AuthAuthority":
        """The authority for *base_url*: display ``endpoint`` and enforceable ``route`` derived from the same URL."""
        return cls(provider=str(provider or ""), endpoint=endpoint_identity(base_url), auth_type=auth_type,
                   auth_source=str(auth_source or "unknown"), entry_id=entry_id, profile=profile,
                   route=route_identity(base_url))

    def describe(self) -> str:
        parts = [f"provider={self.provider or '-'}", f"endpoint={self.endpoint or '-'}",
                 f"auth_type={self.auth_type}", f"auth_source={self.auth_source}"]
        if self.entry_id:
            parts.append(f"entry={self.entry_id}")
        if self.profile:
            parts.append(f"profile={self.profile}")
        return " ".join(parts)

    def as_metadata(self) -> dict:
        """Persistable, secret-free record of the authority (session/delegation metadata)."""
        return {k: v for k, v in (("provider", self.provider), ("endpoint", self.endpoint),
                                  ("auth_type", self.auth_type), ("auth_source", self.auth_source),
                                  ("entry_id", self.entry_id), ("profile", self.profile)) if v}

    def admits(self, entry: Any, base_url: Any) -> bool:
        """Whether pooled *entry* may back this authority for a session now running at *base_url*.

        All of: the same auth type; the same provider (pools are provider-scoped, but a malformed or mixed pool
        must not cross authorities); the session still on the FROZEN route — a mutated ``base_url`` authorizes
        nothing; and an entry that serves that frozen route. Rotation may pick any admitted entry, nothing else.
        Authorities that are not pooled credentials (external process, cloud SDK, no auth) admit none."""
        from agent.credential_pool import credential_pool_entry_serves_endpoint, credential_pool_matches_provider
        if self.auth_type not in (AUTH_OAUTH, AUTH_API_KEY) or getattr(entry, "auth_type", None) != self.auth_type:
            return False
        entry_provider = getattr(entry, "provider", None)
        if entry_provider is not None and not credential_pool_matches_provider(
                str(entry_provider), self.provider, base_url=self.route or None):
            return False
        if route_identity(base_url) != self.route:
            return False
        return credential_pool_entry_serves_endpoint(entry, self.route)
