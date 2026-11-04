"""The candidate profile: versioned storage, `/v1/profile`, and `personalos profile`.

The acceptance criteria this file exists for:

1. **Updates add a version.**
   `test_an_update_adds_a_version_and_the_graph_loads_the_latest` creates a
   profile, updates it, finds both versions intact, and has the job-search
   graph's `load_search_profile` pick up the newer one with no version named.
2. **A job search started over the API loads the stored profile.**
   `test_a_chat_started_job_search_loads_the_actors_stored_profile` writes the
   profile through `/v1/profile`, starts a conversation through `/v1/chat`
   with neither a profile nor a `user_id` in the body, runs the real worker,
   and checks the job board was searched with the stored profile.
3. **The CLI and the API speak one shape.**
   `test_the_cli_and_the_api_round_trip_the_same_profile` writes through one,
   reads through the other, and feeds a read back in as an update.
"""

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apps.api.main import create_app
from apps.api.routes import profiles as profile_routes
from apps.api.routes.profiles import get_profile_store
from apps.api.routes.workflows import build_workflow_services, get_workflow_services
from apps.worker.graphs import build_worker_graphs
from personalos.cli import cli
from personalos.domain.job_search import SearchProfile, WorkMode
from personalos.domain.search_profiles import CandidateProfileInput, actor_user_id
from personalos.persistence.models import Base, CandidateProfileModel, UserModel
from personalos.persistence.search_profiles import SqlSearchProfileStore
from tests.fixtures import job_search_fakes as fakes
from tests.unit.test_job_search_nodes import graph
from tests.unit.test_worker_graphs import _run_worker_until_settled

ACTOR = "maddy"

PROFILE = {
    "target_roles": ["Backend Engineer", "Platform Engineer"],
    "target_locations": ["Berlin", "Remote"],
    "work_modes": ["remote", "hybrid"],
    "seniority_levels": ["senior"],
    "employment_types": ["full_time"],
    "requires_visa_sponsorship": True,
    "constraints": ["no relocation"],
    "keywords": ["python"],
    "must_have_skills": [],
    "excluded_companies": ["Initech"],
    "salary_min": 90000,
    "salary_max": 130000,
    "min_score": 0.6,
    "max_shortlist": 3,
}


@pytest.fixture
def factory(tmp_path: Path) -> sessionmaker:
    engine = create_engine(f"sqlite:///{tmp_path / 'profiles.db'}", connect_args={"timeout": 30})
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def store(factory) -> SqlSearchProfileStore:
    return SqlSearchProfileStore(factory)


@pytest.fixture
def client(factory) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_workflow_services] = lambda: build_workflow_services(factory)
    app.dependency_overrides[get_profile_store] = lambda: SqlSearchProfileStore(factory)
    return TestClient(app, headers={"X-Actor-Id": ACTOR})


def _profile(**changes) -> CandidateProfileInput:
    return CandidateProfileInput.model_validate({**PROFILE, **changes})


# --- Versioning -----------------------------------------------------------------


async def test_an_update_adds_a_version_and_the_graph_loads_the_latest(store, factory):
    user_id = actor_user_id(ACTOR)
    store.create(user_id, _profile())
    store.update(user_id, _profile(target_roles=["Staff Engineer"]))

    versions = store.versions(user_id).versions
    assert [v.profile_version for v in versions] == [2, 1]
    assert versions[0].profile.target_roles == ["Staff Engineer"]
    # The first version is untouched: matches scored against it still mean what they said.
    assert versions[1].profile == _profile()
    assert store.get(user_id, 1).profile == _profile()

    subgraph, _ports = graph(profile_store=store)
    update = await subgraph.load_search_profile({"user_id": str(user_id)})

    loaded = SearchProfile.model_validate(update["search_profile"])
    assert loaded.profile_version == 2
    assert loaded.target_roles == ("Staff Engineer",)
    assert loaded.work_modes == (WorkMode.REMOTE, WorkMode.HYBRID)
    assert loaded.requires_visa_sponsorship is True
    assert loaded.constraints == ("no relocation",)
    assert loaded.remote_only is False

    session = factory()
    try:
        assert session.query(CandidateProfileModel).count() == 2
        assert session.get(UserModel, user_id) is not None
    finally:
        session.close()


async def test_remote_as_the_only_work_mode_makes_the_search_remote_only(store):
    user_id = actor_user_id(ACTOR)
    store.create(user_id, _profile(work_modes=["remote"]))

    assert (await store.load(user_id)).remote_only is True


def test_a_second_create_is_a_conflict_and_an_update_without_a_profile_is_not_found(store):
    from personalos.domain.errors import Conflict, NotFound

    user_id = actor_user_id(ACTOR)
    with pytest.raises(NotFound):
        store.update(user_id, _profile())
    store.create(user_id, _profile())
    with pytest.raises(Conflict):
        store.create(user_id, _profile())


def test_a_profile_written_before_work_modes_reads_back_as_remote(store, factory):
    user_id = actor_user_id(ACTOR)
    session = factory()
    session.add(
        CandidateProfileModel(
            user_id=user_id,
            profile_version=1,
            target_roles=["Analyst"],
            target_locations=[],
            preferences={"remote_only": True, "unknown_key": 1},
        )
    )
    session.commit()
    session.close()

    assert store.get(user_id).profile.work_modes == [WorkMode.REMOTE]


def test_an_actor_that_is_a_uuid_is_its_own_candidate_id():
    assert actor_user_id(str(fakes.USER_ID)) == fakes.USER_ID
    assert actor_user_id(ACTOR) == actor_user_id(ACTOR) != actor_user_id("someone-else")


# --- API ------------------------------------------------------------------------


def test_the_api_creates_reads_updates_and_lists_versions(client):
    created = client.post("/v1/profile", json=PROFILE)
    assert created.status_code == 201, created.text
    assert created.json()["profile_version"] == 1
    assert created.json()["user_id"] == str(actor_user_id(ACTOR))
    assert created.headers["Location"] == "/v1/profile/versions/1"
    assert created.json()["profile"] == PROFILE

    assert client.post("/v1/profile", json=PROFILE).status_code == 409

    changed = {**PROFILE, "target_roles": ["Staff Engineer"]}
    updated = client.put("/v1/profile", json=changed)
    assert updated.status_code == 200, updated.text
    assert updated.json()["profile_version"] == 2

    assert client.get("/v1/profile").json()["profile"] == changed
    assert client.get("/v1/profile/versions/1").json()["profile"] == PROFILE
    listed = client.get("/v1/profile/versions").json()
    assert [v["profile_version"] for v in listed["versions"]] == [2, 1]

    assert client.get("/v1/profile/versions/3").status_code == 404


def test_profiles_are_scoped_to_the_actor(client):
    assert client.post("/v1/profile", json=PROFILE).status_code == 201

    other = client.get("/v1/profile", headers={"X-Actor-Id": "someone-else"})
    assert other.status_code == 404
    assert client.put("/v1/profile", json=PROFILE, headers={"X-Actor-Id": "x"}).status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {**PROFILE, "target_role": ["typo"]},
        {**PROFILE, "work_modes": ["office"]},
        {**PROFILE, "salary_min": 200000, "salary_max": 100000},
        {**PROFILE, "target_roles": [""]},
        {**PROFILE, "min_score": 2},
    ],
    ids=["unknown-field", "unknown-work-mode", "salary-inverted", "blank-role", "score-range"],
)
def test_the_api_refuses_an_invalid_profile(client, body):
    response = client.post("/v1/profile", json=body)

    assert response.status_code == 422, response.text
    assert client.get("/v1/profile").status_code == 404


def test_a_chat_started_job_search_loads_the_actors_stored_profile(client, factory):
    assert client.post("/v1/profile", json=PROFILE).status_code == 201
    changed = {**PROFILE, "target_roles": ["Staff Engineer"], "must_have_skills": ["python"]}
    assert client.put("/v1/profile", json=changed).status_code == 200

    # No profile and no user_id: the actor's own stored profile is the one searched with.
    queued = client.post("/v1/chat", json={"message": "find me a python job"})
    assert queued.status_code == 202, queued.text

    provider = fakes.FakeProvider()
    _run_worker_until_settled(
        factory, build_worker_graphs(factory, overrides={"providers": [provider]})
    )

    assert provider.calls, "the job search never reached the job board"
    searched = provider.calls[0]
    assert searched.user_id == actor_user_id(ACTOR)
    assert searched.profile_version == 2
    assert searched.target_roles == ("Staff Engineer",)
    assert searched.must_have_skills == ("python",)
    assert searched.requires_visa_sponsorship is True


# --- CLI ------------------------------------------------------------------------


def test_the_cli_and_the_api_round_trip_the_same_profile(client, factory, tmp_path, monkeypatch):
    monkeypatch.setattr(profile_routes, "get_profile_store", lambda: SqlSearchProfileStore(factory))
    runner = CliRunner()
    source = tmp_path / "profile.json"
    source.write_text(json.dumps(PROFILE))

    created = runner.invoke(cli, ["profile", "create", str(source), "--actor", ACTOR])
    assert created.exit_code == 0, created.output
    assert json.loads(created.output)["profile"] == PROFILE

    # What the CLI wrote, the API reads back unchanged ...
    from_api = client.get("/v1/profile").json()
    assert from_api["profile"] == PROFILE

    # ... and what the API returns, the CLI takes back as an update.
    edited = {**from_api["profile"], "keywords": ["python", "go"]}
    updated = runner.invoke(cli, ["profile", "update", "--actor", ACTOR], input=json.dumps(edited))
    assert updated.exit_code == 0, updated.output
    assert json.loads(updated.output)["profile_version"] == 2

    shown = runner.invoke(cli, ["profile", "show", "--actor", ACTOR])
    assert shown.exit_code == 0, shown.output
    assert json.loads(shown.output) == client.get("/v1/profile").json()

    first = runner.invoke(cli, ["profile", "show", "--version", "1", "--actor", ACTOR])
    assert json.loads(first.output) == client.get("/v1/profile/versions/1").json()

    listed = runner.invoke(cli, ["profile", "versions", "--actor", ACTOR])
    assert json.loads(listed.output) == client.get("/v1/profile/versions").json()


def test_the_cli_refuses_unknown_fields_and_a_duplicate_create(factory, tmp_path, monkeypatch):
    monkeypatch.setattr(profile_routes, "get_profile_store", lambda: SqlSearchProfileStore(factory))
    runner = CliRunner()

    bad = runner.invoke(cli, ["profile", "create"], input=json.dumps({**PROFILE, "extra": 1}))
    assert bad.exit_code != 0
    assert "extra" in bad.output

    assert runner.invoke(cli, ["profile", "create"], input=json.dumps(PROFILE)).exit_code == 0
    again = runner.invoke(cli, ["profile", "create"], input=json.dumps(PROFILE))
    assert again.exit_code != 0
    assert "already has a profile" in again.output
