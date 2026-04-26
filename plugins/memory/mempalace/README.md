# MemPalace Memory Provider

Local-first persistent memory plugin for Hermes Agent, backed by [MemPalace](https://github.com/your-org/mempalace) (ChromaDB + HNSW).

## Why This Exists

When Hermes compresses its context window to stay within token limits, important
conversation details (decisions, file paths, error resolutions, user preferences)
get summarized and discarded. This plugin hooks into the `on_pre_compress` lifecycle
method to extract and persist that context *before* compression destroys it.

## Features

- **Pre-compression context preservation** — automatically saves key facts from
  messages about to be compressed into the MemPalace
- **Semantic search** — recall past memories across wings/rooms via `mp_search`
- **Turn sync** — important conversation turns are persisted automatically
- **Memory write mirroring** — built-in memory writes are mirrored to the palace
- **Session-end summary** — final context saved when sessions end
- **Idempotent** — deterministic drawer IDs prevent duplicate saves
- **Non-blocking** — all writes happen in background threads

## Configuration

### Install MemPalace

```bash
pip install mempalace-mcp
```

### Initialize a palace

```bash
mempalace init
```

### Enable in Hermes

Add to `~/.hermes/config.yaml`:

```yaml
memory:
  provider: mempalace
mempalace:
  palace_path: ~/.mempalace/palace  # default, optional
```

Or set the environment variable:

```bash
export MEMPALACE_PALACE_PATH=~/.mempalace/palace
```

## How Pre-Compression Works

1. Hermes detects the context window is getting full
2. `MemoryManager.on_pre_compress(messages)` is called with the full message list
3. This plugin extracts key content: user messages, file paths, error resolutions,
   decisions, tool outcomes
4. The extracted content is saved as a drawer in `wing_hermes/pre-compress`
5. Compression proceeds normally — the context is now safely persisted
6. On future turns, `prefetch()` queries the palace and injects relevant past
   context into the system prompt

## Tools

| Tool | Description |
|------|-------------|
| `mp_search` | Semantic search across the palace |
| `mp_file` | File content into wing/room for long-term storage |
| `mp_status` | Check palace health and drawer count |

## Architecture

The plugin uses ChromaDB's `PersistentClient` directly (same approach as the
MemPalace MCP server), with inode-based cache invalidation to detect external
writes. No MCP server process is needed — the plugin talks to the database
directly via the `chromadb` Python package.

## Coexistence with MCP

This plugin can run alongside the MemPalace MCP server. Both access the same
ChromaDB database. The plugin provides automatic lifecycle hooks (pre-compress,
sync_turn) while the MCP server provides the full tool suite (knowledge graph,
diary, tunnels, etc.) to the agent as explicit tools.
