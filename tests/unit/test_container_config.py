"""The compose stack's security posture, checked statically on every test run.

CI's `containers` job checks these against a running stack
(`scripts/smoke_compose.py`); these catch a regression in the files
themselves without Docker. See docs/CONTAINERS.md.
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[2]
COMPOSE = yaml.safe_load((REPO / "docker-compose.yml").read_text())
QUICKSTART = yaml.safe_load((REPO / "compose.quickstart.yml").read_text())
SERVICES = COMPOSE["services"]
DOCKERFILE = (REPO / "Dockerfile").read_text()
UI_DOCKERFILE = (REPO / "ui" / "Dockerfile").read_text()

#: `name:tag@sha256:<64 hex>`: a pinned third-party image.
PINNED = re.compile(r"^[\w./-]+:[\w.-]+@sha256:[0-9a-f]{64}$")
#: Images this repository builds.
LOCAL_IMAGE = re.compile(r"^personalos-[\w-]+:\$\{PERSONALOS_IMAGE_TAG:-local\}$")

APP_SERVICES = ("migrate", "api", "worker")


def _ports(service: dict) -> list[str]:
    return [str(p) for p in service.get("ports", [])]


def test_the_stack_has_every_required_service():
    assert {"postgres", "migrate", "api", "worker", "otel-collector"} <= set(SERVICES)
    assert SERVICES["ollama"]["profiles"] == ["ollama"]
    assert SERVICES["ui"]["profiles"] == ["ui"]


def test_postgres_has_no_host_port_and_only_the_internal_network():
    postgres = SERVICES["postgres"]

    assert "ports" not in postgres
    assert "expose" not in postgres
    assert postgres["networks"] == ["backend"]
    assert COMPOSE["networks"]["backend"]["internal"] is True


def test_postgres_has_no_password_in_the_compose_file():
    environment = SERVICES["postgres"]["environment"]

    assert "POSTGRES_PASSWORD" not in environment
    assert environment["POSTGRES_PASSWORD_FILE"] == "/run/personalos-db/password"


@pytest.mark.parametrize(
    "source", [COMPOSE, QUICKSTART], ids=["docker-compose.yml", "compose.quickstart.yml"]
)
def test_every_published_port_is_on_loopback(source):
    for name, service in source["services"].items():
        for port in _ports(service):
            assert port.startswith("127.0.0.1:"), f"{name} publishes {port}"


def test_only_the_api_and_ui_publish_ports():
    published = {name for name, service in SERVICES.items() if _ports(service)}

    assert published == {"api", "ui"}


def test_the_api_publishes_its_port_on_loopback():
    assert _ports(SERVICES["api"]) == ["127.0.0.1:${PERSONALOS_API_PORT:-8000}:8000"]


def test_the_collector_and_ollama_have_no_host_port():
    assert "ports" not in SERVICES["otel-collector"]
    assert "ports" not in SERVICES["ollama"]


def test_third_party_images_are_pinned_by_tag_and_digest():
    for name, service in SERVICES.items():
        if "build" in service:
            image = service.get("image")
            assert image is None or LOCAL_IMAGE.match(image), f"{name}: {image}"
        else:
            assert PINNED.match(service["image"]), f"{name} is not pinned: {service['image']}"


@pytest.mark.parametrize("dockerfile", [DOCKERFILE, UI_DOCKERFILE], ids=["Dockerfile", "ui"])
def test_dockerfile_base_images_are_pinned_by_tag_and_digest(dockerfile):
    bases = re.findall(r"^ARG \w+_IMAGE=(\S+)$", dockerfile, re.MULTILINE)

    assert bases
    for base in bases:
        assert PINNED.match(base), base
    for line in re.findall(r"^FROM (\S+)", dockerfile, re.MULTILINE):
        # Every FROM is a pinned ARG or an earlier stage, never a bare tag.
        assert line.startswith("${") or line in {"runtime", "worker"}, line


def test_the_app_images_run_as_a_non_root_user():
    # Every stage that ends up as an image switches back to the app user.
    assert "USER 10001:10001" in DOCKERFILE
    users = re.findall(r"^USER (\S+)$", DOCKERFILE, re.MULTILINE)
    assert users[-1] == "10001:10001"
    assert re.findall(r"^USER (\S+)$", UI_DOCKERFILE, re.MULTILINE)[-1] == "101"


def test_the_app_containers_are_hardened():
    for name in (*APP_SERVICES, "db-password", "ui", "otel-collector"):
        service = SERVICES[name]
        assert service.get("read_only") is True, name
        assert service.get("cap_drop") == ["ALL"], name
        assert "no-new-privileges:true" in service.get("security_opt", []), name


def test_the_image_build_context_is_an_allow_list():
    lines = [
        line.strip()
        for line in (REPO / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

    assert lines[0] == "*"
    allowed = {line[1:].rstrip("/") for line in lines if line.startswith("!")}
    assert "tests" not in allowed and "evals" not in allowed
    assert not any(entry.startswith(".env") for entry in allowed)


def test_the_images_refuse_dev_and_test_dependencies_and_drop_the_fakes():
    assert 'if extra in ("dev", "test"):' in DOCKERFILE
    assert "rm -f apps/worker/fake_providers.py personalos/providers/fake.py" in DOCKERFILE


#: A setting that names a secret, as opposed to where one is kept.
SECRET_NAME = re.compile(r"PASSWORD|TOKEN|SECRET|API_KEY")
WHERE_NOT_WHAT = ("_FILE", "_STORE", "_DIR", "_DIRS", "_BACKEND", "_REF", "_ENV")


def _names_a_secret_value(name: str) -> bool:
    return bool(SECRET_NAME.search(name)) and not name.endswith(WHERE_NOT_WHAT)


def test_no_secret_is_set_in_an_image():
    for dockerfile in (DOCKERFILE, UI_DOCKERFILE):
        for name in re.findall(r"\b([A-Z][A-Z0-9_]*)=", dockerfile):
            assert not _names_a_secret_value(name), f"the Dockerfile sets {name}"


def test_no_secret_is_set_in_the_compose_files():
    for source in (COMPOSE, QUICKSTART):
        for service_name, service in source["services"].items():
            for name in service.get("environment") or {}:
                assert not _names_a_secret_value(name), f"{service_name} sets {name}"
    for env_file in ("deploy/compose.defaults.env",):
        for line in (REPO / env_file).read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                name = line.split("=", 1)[0].strip()
                assert not _names_a_secret_value(name), f"{env_file} sets {name}"


def test_the_containers_use_the_file_secret_store_and_keep_references():
    environment = SERVICES["api"]["environment"]

    assert environment["SECRET_STORE_BACKEND"] == "file"
    assert environment["SECRET_STORE_DIR"] == "/var/lib/personalos/secrets"
    assert environment["SECRET_STORE_READ_ONLY_DIRS"] == '["/run/secrets"]'
    assert "SECRET_STORE_BACKEND=file" in DOCKERFILE


def test_health_checks_exist_for_everything_that_can_have_one():
    assert "healthcheck" in SERVICES["postgres"]
    assert "healthcheck" in SERVICES["ollama"]
    # The app and UI images carry their own.
    assert DOCKERFILE.count("HEALTHCHECK") == 2
    assert "worker-health" in DOCKERFILE
    assert "/health" in DOCKERFILE
    assert "HEALTHCHECK" in UI_DOCKERFILE


def _depends(name: str) -> dict:
    return SERVICES[name]["depends_on"]


def test_the_api_and_worker_wait_for_migrations_and_a_healthy_database():
    for name in ("api", "worker"):
        depends = _depends(name)
        assert depends["migrate"]["condition"] == "service_completed_successfully", name
        assert depends["postgres"]["condition"] == "service_healthy", name


def test_migrations_run_once_against_a_healthy_database():
    migrate = SERVICES["migrate"]

    assert migrate["command"] == ["alembic", "upgrade", "head"]
    assert migrate["restart"] == "no"
    assert _depends("migrate")["postgres"]["condition"] == "service_healthy"
    assert _depends("postgres")["db-password"]["condition"] == "service_completed_successfully"


def test_the_ui_waits_for_a_healthy_api():
    assert _depends("ui")["api"]["condition"] == "service_healthy"


def test_state_lives_in_named_volumes():
    volumes = set(COMPOSE["volumes"])

    assert {"postgres-data", "ollama-models", "personalos-secrets", "db-password"} <= volumes
    assert "postgres-data:/var/lib/postgresql/data" in SERVICES["postgres"]["volumes"]
    assert "ollama-models:/root/.ollama" in SERVICES["ollama"]["volumes"]


def test_the_quickstart_override_only_swaps_the_worker():
    assert set(QUICKSTART["services"]) == {"worker"}
    worker = QUICKSTART["services"]["worker"]
    assert worker["build"]["target"] == "worker-quickstart"
    assert worker["environment"]["WORKER_GRAPHS"] == (
        "apps.worker.fake_providers:build_worker_graphs"
    )


def test_the_collector_config_is_mounted_and_receives_otlp():
    collector = yaml.safe_load((REPO / "deploy" / "otel" / "collector.yaml").read_text())

    assert collector["receivers"]["otlp"]["protocols"]["http"]["endpoint"].endswith(":4318")
    assert "health_check" in collector["service"]["extensions"]
    assert SERVICES["api"]["environment"]["OTEL_EXPORTER_OTLP_ENDPOINT"] == (
        "http://otel-collector:4318"
    )
