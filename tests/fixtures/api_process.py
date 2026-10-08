"""An API process that serves one request and exits.

Exists for the restart test in `tests/unit/test_workflow_routes.py`: a workflow
is started by *this* process, which then exits, and a different process asks
for its status. Anything the first process kept in memory is gone by then, so
a correct answer can only have come from the database.

It runs the real wiring, not a test override: `DATABASE_URL` points
`personalos.persistence.database.SessionLocal` at the test's database, and the
request goes through `create_app()` with its middleware and the default
`get_workflow_services` dependency.

Run as `DATABASE_URL=sqlite:///<db> python -m tests.fixtures.api_process
<method> <path> <json-body|-> [<header>=<value> ...]`; prints the status code
and the JSON body, one per line.
"""

import json
import sys

from fastapi.testclient import TestClient

from apps.api.main import create_app


def main(argv: list[str]) -> None:
    method, path, raw_body, *header_args = argv
    body = None if raw_body == "-" else json.loads(raw_body)
    headers = dict(arg.split("=", 1) for arg in header_args)

    # No `with`: the lifespan (table creation, MCP registration) is the
    # deployment's startup, not part of serving a request, and the parent has
    # already created the schema.
    client = TestClient(create_app())
    response = client.request(method, path, json=body, headers=headers)
    print(response.status_code)
    print(json.dumps(response.json()))


if __name__ == "__main__":
    main(sys.argv[1:])
