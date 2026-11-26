"""The container secret store: files named by reference, behind the same `SecretStore` port.

A container has no OS keychain, so `FileSecretStore` keeps secrets as files:
written `0600` into a writable directory, and read from read-only mounts
(Docker secrets). These pin down what makes that safe -- a reference maps to
one file name that cannot escape the directory, a loosely permissioned file
is refused, a read-only mount is never written -- and that the composition
root chooses it only when told to.
"""

import logging
import os
import stat

import pytest

from personalos import bootstrap
from personalos.config import Settings, settings
from personalos.domain.credentials import CredentialRef, SecretValue
from personalos.secrets import FileSecretStore, SecretStore, SecretStoreUnavailable
from personalos.secrets.api_token import API_TOKEN_REF, KeychainApiTokenStore
from personalos.secrets.store import KeyringSecretStore, secret_file_name

GMAIL = CredentialRef(provider="google", name="alice@example.com")
CLIENT = CredentialRef(provider="google-oauth-client", name="default")
REFRESH_TOKEN = "1//refresh-token-value"

posix_only = pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX")


def test_it_implements_the_secret_store_port(tmp_path):
    assert isinstance(FileSecretStore(tmp_path), SecretStore)


def test_a_reference_names_one_flat_file():
    assert secret_file_name(GMAIL) == "google.alice@example.com"
    assert secret_file_name(CLIENT) == "google-oauth-client.default"
    assert secret_file_name(API_TOKEN_REF) == "personalos-api.local"


def test_secrets_round_trip_by_reference(tmp_path):
    store = FileSecretStore(tmp_path / "secrets")

    assert store.get(GMAIL) is None
    store.put(GMAIL, SecretValue(REFRESH_TOKEN))

    assert store.get(GMAIL) == SecretValue(REFRESH_TOKEN)
    assert store.get(CLIENT) is None
    store.delete(GMAIL)
    assert store.get(GMAIL) is None
    store.delete(GMAIL)  # deleting what is gone is a no-op


@posix_only
def test_secrets_are_written_owner_only(tmp_path):
    store = FileSecretStore(tmp_path / "secrets")
    store.put(GMAIL, SecretValue(REFRESH_TOKEN))

    directory = tmp_path / "secrets"
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE((directory / "google.alice@example.com").stat().st_mode) == 0o600
    # Atomic replace leaves no temporary file behind.
    assert [p.name for p in directory.iterdir()] == ["google.alice@example.com"]


@posix_only
def test_a_secret_file_others_can_read_is_refused(tmp_path):
    store = FileSecretStore(tmp_path)
    store.put(GMAIL, SecretValue(REFRESH_TOKEN))
    os.chmod(tmp_path / secret_file_name(GMAIL), 0o644)

    with pytest.raises(SecretStoreUnavailable, match="accessible to other users"):
        store.get(GMAIL)


def test_docker_secrets_are_read_from_a_read_only_directory(tmp_path):
    mounted = tmp_path / "run-secrets"
    mounted.mkdir()
    # Docker controls a mounted secret's mode, so a read-only dir is not mode-checked.
    (mounted / "google-oauth-client.default").write_text("client-secret\n")
    store = FileSecretStore(tmp_path / "secrets", read_only_dirs=[mounted])

    assert store.get(CLIENT) == SecretValue("client-secret")


def test_the_writable_directory_wins_over_a_mount(tmp_path):
    mounted = tmp_path / "run-secrets"
    mounted.mkdir()
    (mounted / secret_file_name(GMAIL)).write_text("first-token")
    store = FileSecretStore(tmp_path / "secrets", read_only_dirs=[mounted])

    store.put(GMAIL, SecretValue("rotated-token"))

    assert store.get(GMAIL) == SecretValue("rotated-token")
    assert (mounted / secret_file_name(GMAIL)).read_text() == "first-token"


def test_delete_never_touches_a_read_only_mount(tmp_path, caplog):
    mounted = tmp_path / "run-secrets"
    mounted.mkdir()
    (mounted / secret_file_name(CLIENT)).write_text("client-secret")
    store = FileSecretStore(tmp_path / "secrets", read_only_dirs=[mounted])

    with caplog.at_level(logging.WARNING, logger="personalos.secrets.store"):
        store.delete(CLIENT)

    assert (mounted / secret_file_name(CLIENT)).exists()
    assert "read-only mount" in caplog.text


def test_an_empty_file_holds_no_secret(tmp_path):
    (tmp_path / secret_file_name(GMAIL)).write_text("\n")
    os.chmod(tmp_path / secret_file_name(GMAIL), 0o600)

    assert FileSecretStore(tmp_path).get(GMAIL) is None


def test_a_directory_where_a_secret_should_be_is_refused(tmp_path):
    (tmp_path / secret_file_name(GMAIL)).mkdir()

    with pytest.raises(SecretStoreUnavailable, match="not a regular file"):
        FileSecretStore(tmp_path).get(GMAIL)


@posix_only
def test_an_unwritable_directory_fails_closed_without_quoting_the_secret(tmp_path):
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    if os.access(locked, os.W_OK):  # running as root: permissions do not bind
        pytest.skip("directory permissions are not enforced for this user")

    with pytest.raises(SecretStoreUnavailable) as raised:
        FileSecretStore(locked).put(GMAIL, SecretValue(REFRESH_TOKEN))

    assert REFRESH_TOKEN not in str(raised.value)


def test_storing_a_secret_does_not_log_it(tmp_path, caplog):
    with caplog.at_level(logging.DEBUG):
        FileSecretStore(tmp_path).put(GMAIL, SecretValue(REFRESH_TOKEN))

    assert REFRESH_TOKEN not in caplog.text
    assert str(GMAIL) in caplog.text


def test_the_api_token_can_live_in_the_file_store(tmp_path):
    token_store = KeychainApiTokenStore(FileSecretStore(tmp_path))
    token_store.write(SecretValue("pos_at_example"))

    assert token_store.read() == SecretValue("pos_at_example")
    assert (tmp_path / "personalos-api.local").exists()


# --- Which store the composition root builds ----------------------------------


def test_the_keychain_stays_the_default():
    assert Settings(_env_file=None).secret_store_backend == "keychain"


def test_the_composition_root_builds_the_keychain_store_by_default(monkeypatch):
    built = []
    monkeypatch.setattr(settings, "secret_store_backend", "keychain")
    monkeypatch.setattr(
        bootstrap, "KeyringSecretStore", lambda service: built.append(service) or "keychain"
    )

    assert bootstrap.build_secret_store() == "keychain"
    assert built == [settings.secret_store_service]


def test_the_composition_root_builds_the_file_store_when_told_to(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "secret_store_backend", "file")
    monkeypatch.setattr(settings, "secret_store_dir", str(tmp_path / "secrets"))
    monkeypatch.setattr(settings, "secret_store_read_only_dirs", ["/run/secrets"])

    store = bootstrap.build_secret_store()

    assert isinstance(store, FileSecretStore)
    assert store.directory == tmp_path / "secrets"
    assert [str(d) for d in store.read_only_dirs] == ["/run/secrets"]


def test_the_file_store_needs_a_directory(monkeypatch):
    monkeypatch.setattr(settings, "secret_store_backend", "file")
    monkeypatch.setattr(settings, "secret_store_dir", "")

    with pytest.raises(SecretStoreUnavailable, match="SECRET_STORE_DIR"):
        bootstrap.build_secret_store()


def test_an_unknown_backend_is_refused_rather_than_guessed(monkeypatch):
    monkeypatch.setattr(settings, "secret_store_backend", "plaintext")

    with pytest.raises(SecretStoreUnavailable, match="keychain' or 'file"):
        bootstrap.build_secret_store()


def test_a_broken_keychain_does_not_fall_back_to_files(monkeypatch, tmp_path):
    def unavailable(service):
        raise SecretStoreUnavailable("no keychain")

    monkeypatch.setattr(settings, "secret_store_backend", "keychain")
    monkeypatch.setattr(settings, "secret_store_dir", str(tmp_path))
    monkeypatch.setattr(bootstrap, "KeyringSecretStore", unavailable)

    with pytest.raises(SecretStoreUnavailable, match="no keychain"):
        bootstrap.build_secret_store()
    assert KeyringSecretStore is not None  # the real class is untouched
    assert list(tmp_path.iterdir()) == []


# --- The database password file -----------------------------------------------


def test_the_database_password_file_replaces_the_urls_password(tmp_path):
    password_file = tmp_path / "password"
    password_file.write_text("p@ss/w%rd\n")

    configured = Settings(
        _env_file=None,
        database_url="postgresql://personalos:placeholder@postgres:5432/personalos",
        database_password_file=str(password_file),
    )

    from sqlalchemy.engine import make_url

    url = make_url(configured.database_url)
    assert url.password == "p@ss/w%rd"
    assert (url.username, url.host, url.port, url.database) == (
        "personalos",
        "postgres",
        5432,
        "personalos",
    )
    assert url.drivername == "postgresql+psycopg2"


def test_a_missing_database_password_file_fails_at_startup(tmp_path):
    with pytest.raises(ValueError, match="DATABASE_PASSWORD_FILE"):
        Settings(
            _env_file=None,
            database_url="postgresql://personalos@postgres/personalos",
            database_password_file=str(tmp_path / "absent"),
        )


def test_an_empty_database_password_file_is_refused(tmp_path):
    (tmp_path / "password").write_text("\n")

    with pytest.raises(ValueError, match="is empty"):
        Settings(
            _env_file=None,
            database_url="postgresql://personalos@postgres/personalos",
            database_password_file=str(tmp_path / "password"),
        )
