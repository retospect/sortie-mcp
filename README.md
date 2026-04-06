# sortie-mcp

Campaign orchestration MCP server for AI agents — dependency DAGs, parallel
fan-out, failure policies, and embedded notes.

Think `make` for AI agent workflows, where the LLM is the planner that
generates and adapts the DAG at runtime.

## Install

```bash
pip install sortie-mcp
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add sortie-mcp
```

## Quick Start

### 1. Set up PostgreSQL

sortie-mcp requires PostgreSQL 15+ with [pgvector](https://github.com/pgvector/pgvector).

```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/mydb"
export SORTIE_SCHEMA="sortie"  # default
```

### 2. Run the MCP server

```bash
sortie-mcp
# or: python -m sortie_mcp.server
```

The server runs on stdio transport. Configure it in your MCP client:

```json
{
  "sortie": {
    "command": ["sortie-mcp"],
    "env": {
      "DATABASE_URL": "postgresql://..."
    }
  }
}
```

### 3. Run the campaign runner

```bash
sortie-runner
# or: python -m sortie_mcp.runner
```

Add to cron for autonomous operation:

```
*/15 * * * * /path/to/venv/bin/sortie-runner
```

## Architecture

One MCP server, three perspectives:

- **Coordinator** (e.g. a dispatcher agent): create, list, steer, pause/cancel campaigns
- **Worker** (specialist agents): get context, add notes, complete/fail steps, spawn subtasks
- **Runner** (cron): capacity-aware watchdog that dispatches ready steps and consults the planner LLM

### Step Types

| Type | Description |
|------|-------------|
| `atomic` | Single task executed by one agent |
| `parallel_group` | Fan-out: children run concurrently |
| `sequence` | Pipeline: each step depends on the previous |
| `for_each` | Map: apply a template to each item in a list |

### Key Features

- **DAG splice** (`spawn_and_continue`): agents can split work into subtasks + continuation
- **Branch abort** (`abort_branch`): scoped early return from an ancestor step
- **Skip cascade**: transitive propagation through the dependency graph
- **Priority scheduling**: urgent / high / normal / low / background
- **Advisory dedup**: fingerprinting warns the planner of duplicate steps
- **Depth limits**: `spawn_and_continue` hidden from agents at max depth
- **Embedded notes**: pgvector semantic search across campaign findings

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `DATABASE_URL` | `postgresql://localhost/sortie` | PostgreSQL connection string |
| `SORTIE_SCHEMA` | `sortie` | Database schema name |
| `SORTIE_MAX_CONCURRENT` | `4` | Max parallel running steps |
| `SORTIE_ZOMBIE_TIMEOUT` | `30` | Minutes before a stuck step is reset |
| `LITELLM_URL` | `http://localhost:4000` | LiteLLM proxy URL (for planner) |
| `LITELLM_KEY` | | LiteLLM API key |
| `SORTIE_PLANNER_MODEL` | `qwen3.5:9b` | Model for the planner LLM |
| `OPENCLAW_RUNTIME_URL` | `http://localhost:3000` | Agent runtime API |

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run mypy src tests
```

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).
