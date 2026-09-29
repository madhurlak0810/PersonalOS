"""Tests for settings normalization, chiefly the database URL's driver.

The case these exist for is a real break rather than a hypothetical: SQLAlchemy
2.1 changed which DBAPI a bare `postgresql://` URL resolves to (psycopg2 ->
psycopg v3). Because `personalos.persistence.database` calls `create_engine` at
import time, and `create_engine` imports the DBAPI to load the dialect, a bare
URL stopped being importable at all -- and it took down every test module that
transitively imports the persistence package, fourteen of them, as collection
errors rather than as a legible failure.

`test_the_named_driver_is_actually_installed` and
`test_create_engine_succeeds_on_a_bare_postgres_url` are the two that would have
caught it. They are deliberately about the *driver being importable*, not about
any particular SQLAlchemy version, so they keep holding if the default moves
again.
"""

import importlib

import pytest
from sqlalchemy import create_engine

from personalos.config import POSTGRES_DRIVER, Settings


def test_a_bare_postgres_url_is_rewritten_to_name_the_driver():
    settings = Settings(database_url="postgresql://user:pw@localhost:5432/personalos")

    assert settings.database_url == (
        f"postgresql+{POSTGRES_DRIVER}://user:pw@localhost:5432/personalos"
    )


def test_the_default_url_already_names_the_driver():
    assert Settings().database_url.startswith(f"postgresql+{POSTGRES_DRIVER}://")


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+psycopg2://user:pw@localhost/db",
        "postgresql+psycopg://user:pw@localhost/db",
        "postgresql+asyncpg://user:pw@localhost/db",
    ],
)
def test_a_url_that_already_names_a_driver_is_left_exactly_as_written(url):
    """Naming a driver is an explicit choice; overriding it would be surprising."""
    assert Settings(database_url=url).database_url == url


@pytest.mark.parametrize(
    "url", ["sqlite://", "sqlite:///./local.db", "mysql+pymysql://user:pw@localhost/db"]
)
def test_a_non_postgres_url_is_untouched(url):
    assert Settings(database_url=url).database_url == url


def test_only_the_scheme_is_rewritten_not_a_matching_substring_elsewhere():
    """The rewrite is anchored at the scheme, so a password or host is safe."""
    url = "postgresql://user:postgresql://@localhost:5432/db"

    assert Settings(database_url=url).database_url == (
        f"postgresql+{POSTGRES_DRIVER}://user:postgresql://@localhost:5432/db"
    )


def test_the_named_driver_is_actually_installed():
    """The driver the URL names has to be importable, or nothing can connect.

    This is the assertion that turns "a dependency changed a default" into one
    readable failure instead of fourteen collection errors.
    """
    assert importlib.import_module(POSTGRES_DRIVER)


def test_create_engine_succeeds_on_a_bare_postgres_url():
    """The exact operation that broke: build an engine, no server needed.

    `create_engine` does not connect, but it does import the DBAPI to load the
    dialect -- which is why an uninstalled driver fails here, at import time,
    rather than later at first query.
    """
    engine = create_engine(
        Settings(database_url="postgresql://user:pw@localhost:5432/personalos").database_url
    )

    assert engine.dialect.name == "postgresql"
    assert engine.dialect.driver == POSTGRES_DRIVER


def test_the_persistence_engine_uses_the_installed_driver():
    """The module-level engine every repository binds to is the one that must work."""
    from personalos.persistence.database import engine

    if engine.dialect.name != "postgresql":
        pytest.skip("DATABASE_URL does not point at postgresql in this environment")
    assert engine.dialect.driver == POSTGRES_DRIVER
