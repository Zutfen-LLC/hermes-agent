"""Construction-local init-time guard for the restricted tool-free envelope (R3).

Holds the restricted binding for EXACTLY one ``AIAgent`` construction:
``gateway.platforms.api_server_restricted_runs._new_restricted_agent`` sets it on a
private ``contextvars.Context`` and runs the constructor inside it, and the
client-construction chokepoint (``agent.agent_runtime_helpers.create_openai_client``)
reads it only for an agent that has no durable per-instance binding yet — i.e. its own
initialization-time client build.

Isolation properties (why this replaced the temporary class attribute):
- The variable lives on the construction's own ``Context`` (``copy_context().run``), so
  a concurrently constructed agent — ordinary or restricted — observes only its own
  context state; no other request can see, overwrite, or delete this binding.
- There is no mutable class/module state to clean up: constructor exceptions cannot
  leak a guard to a sibling or drop one a sibling is still reading.
- This module is intentionally dependency-free (stdlib only) so both the gateway layer
  and the agent-layer chokepoint can import it without cycle or outage coupling.
"""

import contextvars

_restricted_init_binding = contextvars.ContextVar("_restricted_init_binding", default=None)
