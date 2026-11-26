"""Checks run *inside* a built API or worker image; exits non-zero on the first failure.

    docker run --rm -i --entrypoint python personalos-api:local - < scripts/check_image.py

Not copied into the image (scripts/ is outside the build context's
allow-list); CI pipes it in. Checks what the Dockerfile promises:

- it runs as a non-root user;
- no test fakes, no tests, and no dev or test dependencies are installed;
- nothing is in the secret store directory, and no `.env` is anywhere in /app;
- the API, the worker and the alert receiver still import without the fakes.
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

FAKES = ("/app/apps/worker/fake_providers.py", "/app/personalos/providers/fake.py")
DEV_PACKAGES = ("pytest", "faker", "ruff", "black", "mypy", "pre_commit", "pytest_asyncio")
SECRET_DIRS = ("/var/lib/personalos/secrets", "/var/lib/personalos/db")


def problems() -> list[str]:
    found = []
    if os.getuid() == 0:
        found.append("runs as root")
    found += [f"test fake present: {path}" for path in FAKES if Path(path).exists()]
    if Path("/app/tests").exists():
        found.append("tests/ is in the image")
    found += [
        f"dev dependency installed: {name}"
        for name in DEV_PACKAGES
        if importlib.util.find_spec(name) is not None
    ]
    for directory in SECRET_DIRS:
        path = Path(directory)
        if path.exists() and any(path.iterdir()):
            found.append(f"{directory} is not empty")
    found += [f"env file in the image: {path}" for path in Path("/app").rglob(".env*")]
    for module in (
        "apps.api.main",
        "apps.worker.main",
        "apps.worker.graphs",
        "apps.alerts.receiver",
    ):
        try:
            importlib.import_module(module)
        except Exception as error:  # noqa: BLE001 - any import failure is the finding
            found.append(f"{module} does not import: {type(error).__name__}: {error}")
    return found


if __name__ == "__main__":
    failures = problems()
    for failure in failures:
        print(f"IMAGE CHECK FAILED: {failure}", file=sys.stderr)
    if not failures:
        print(f"image checks passed (uid {os.getuid()})")
    sys.exit(1 if failures else 0)
