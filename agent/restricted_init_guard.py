"""Construction-local init-time guard for the restricted tool-free envelope (R3).

Holds the restricted binding for EXACTLY one ``AIAgent`` construction:
``gateway.platforms.api_server_restricted_runs._new_restricted_agent`` sets it on a
private ``contextvars.Context`` and runs the constructor inside it, and the
client-construction chokepoint (``agent.agent_runtime_helpers.create_openai_client``)
reads it only for an agent that has no durable per-instance binding yet — i.e. its own
initialization-time client build. The carried binding tuple's fourth element is the
restricted wire dialect (``chat_completions`` or, since #35, ``codex_responses``),
taken from the resolved route rather than hardcoded, so each dialect's guard binds its
own wire identity.

Isolation properties (why this replaced the temporary class attribute):
- The variable lives on the construction's own ``Context`` (``copy_context().run``), so
  a concurrently constructed agent — ordinary or restricted — on another thread or
  task observes only its own context state; no other request can see, overwrite, or
  delete this binding.
- The carried value pairs the binding with a ONE-SHOT construction token minted by
  ``restricted_construction()`` and installed on the guarded instance by
  ``claim_construction``. A NESTED construction running inside the guarded
  constructor's dynamic extent (e.g. triggered from a hostile provider hook) sees the
  ContextVar but cannot claim it: its instance does not carry the matching token, so
  the nested agent builds unguarded exactly like any ordinary agent, and it cannot
  consume or disable the parent's guard.
- There is no mutable class/module state to clean up: constructor exceptions cannot
  leak a guard to a sibling or drop one a sibling is still reading.
- This module is intentionally dependency-free (stdlib only) so both the gateway layer
  and the agent-layer chokepoint can import it without cycle or outage coupling.
"""

import contextvars
import itertools
from typing import Any

_construction_seq = itertools.count()


class RestrictedConstruction:
    """Single-use token binding one ContextVar entry to exactly one AIAgent instance."""

    __slots__ = ("seq",)

    def __init__(self, seq: int) -> None:
        self.seq = seq


def restricted_construction() -> RestrictedConstruction:
    """Mint the token for one restricted construction (called once per construction)."""
    return RestrictedConstruction(next(_construction_seq))


def claim_construction(instance: Any, token: RestrictedConstruction) -> None:
    """Install the construction token on the instance being constructed.

    Called by the guarded factory on the object ``AIAgent.__new__`` just produced;
    the chokepoint accepts the ContextVar binding only from an instance whose token
    matches the one the ContextVar carries (single-use).
    """
    try:
        instance._restricted_construction_token = token
    except AttributeError:  # pragma: no cover - AIAgent instances accept attributes
        pass


_restricted_init_binding: contextvars.ContextVar[tuple | None] = contextvars.ContextVar(
    "_restricted_init_binding", default=None)
