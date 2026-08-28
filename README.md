# PersonalOS Agent

Repository skeleton for the PersonalOS local-first agentic assistant.

This repository currently contains structure only. Implementation will follow the architecture and engineering specification supplied for the project.

## Planned Areas

- `apps/`: API gateway and background worker entrypoints
- `personalos/`: application packages grouped by workflow and ownership boundary
- `mcp_servers/`: capability-specific MCP server packages
- `tests/`: unit, graph scenario, adversarial, and fixture material
- `evals/`: golden datasets and model benchmark material
- `migrations/`: PostgreSQL schema migrations

No runtime code, dependencies, database configuration, or agent logic has been added yet.
