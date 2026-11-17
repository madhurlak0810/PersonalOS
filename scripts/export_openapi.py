"""Write the API's OpenAPI schema, the contract the UI's typed client is built from.

    python scripts/export_openapi.py            # rewrite ui/openapi.json
    python scripts/export_openapi.py --check    # non-zero if ui/openapi.json is stale
    python scripts/export_openapi.py --stdout   # print it instead

`ui/src/api/schema.d.ts` is generated from `ui/openapi.json` by
`openapi-typescript` (`npm run api:generate` in `ui/`), so a route or model
change reaches the UI's types in two steps: this script, then that one. CI runs
both checks, so neither file can drift from the code behind it.

The schema is built from `apps.api.main.create_app()` without serving
anything: no database, no token, no network.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Where the UI reads the schema from.
SCHEMA_PATH = REPO / "ui" / "openapi.json"


def render() -> str:
    """The schema as the file holds it: stable key order, trailing newline."""
    from apps.api.main import create_app

    return json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if the file is out of date")
    mode.add_argument("--stdout", action="store_true", help="print instead of writing")
    args = parser.parse_args(argv)

    schema = render()
    if args.stdout:
        sys.stdout.write(schema)
        return 0
    if args.check:
        current = SCHEMA_PATH.read_text() if SCHEMA_PATH.exists() else ""
        if current != schema:
            print(
                f"{SCHEMA_PATH.relative_to(REPO)} is out of date with the API; run "
                "`python scripts/export_openapi.py` and then `npm run api:generate` in ui/",
                file=sys.stderr,
            )
            return 1
        print(f"{SCHEMA_PATH.relative_to(REPO)} matches the API")
        return 0
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(schema)
    print(f"wrote {SCHEMA_PATH.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
