"""The authentication authority a session runs on: provider, endpoint, auth mechanism and credential source.

An ``AuthAuthority`` names WHICH credential authority backs a route without carrying the credential itself, so it
can travel into logs, errors and session metadata. Delegated children bind to one at construction and keep it for
their lifetime: credential rotation may move between entries of the same authority, never to another auth type or
endpoint (ops-supervisor#216).

The auth mechanism a provider mandates comes from its registered metadata (``PROVIDER_REGISTRY`` rows, which
provider plugins mirror their ``ProviderProfile.auth_type`` into), never from provider or model names.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def provider_auth_mechanism(provider: Any) -> Optional[str]:
    """The one auth mechanism *provider*'s registered metadata mandates, or None when it accepts several."""
    if not isinstance(provider, str) or not provider:
        return None
    from hermes_cli.auth_plugin_providers import registry_lookup
    from providers import get_provider_profile
    # Aliases (``codex`` → ``openai-codex``) resolve through the provider registry, like every other lookup.
    canonical = getattr(get_provider_profile(provider.strip().lower()), "name", None) or provider.strip().lower()
    registry_auth_type = str(getattr(registry_lookup(canonical), "auth_type", "") or "")
    if registry_auth_type.startswith("oauth"):
        return AUTH_OAUTH
    return _REGISTRY_MECHANISMS.get(registry_auth_type)


def endpoint_identity(base_url: Any) -> str:
    """Non-secret, normalized endpoint identity (host + path; userinfo and query never included)."""
    from urllib.parse import urlsplit
    from hermes_cli.route_identity import normalize_route_base_url
    normalized = normalize_route_base_url(base_url)
    try:
        parts = urlsplit(normalized)
    except ValueError:
        return "-"
    if not parts.hostname:
        return "-"
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.hostname}{port}{parts.path.rstrip('/')}"


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
        """Whether pooled *entry* belongs to this authority for a session running at *base_url*: same auth type
        and an entry that serves that endpoint. Rotation may pick any admitted entry, nothing else; authorities
        that are not pooled credentials (external process, cloud SDK, no auth) admit none."""
        from agent.credential_pool import credential_pool_entry_serves_endpoint
        if self.auth_type not in (AUTH_OAUTH, AUTH_API_KEY) or getattr(entry, "auth_type", None) != self.auth_type:
            return False
        return credential_pool_entry_serves_endpoint(entry, base_url)
