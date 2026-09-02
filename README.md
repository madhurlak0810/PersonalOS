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
- `docs/`: architecture and design documentation

## Architecture

Ownership boundaries between orchestration, policy, execution, persistence and
tool adapters are defined in
[docs/ARCHITECTURE_BOUNDARIES.md](docs/ARCHITECTURE_BOUNDARIES.md) and enforced
in CI:

```bash
python scripts/check_boundaries.py
pytest tests/architecture
```

No runtime code, dependencies, database configuration, or agent logic has been added yet.
