"""The worker's liveness file and the `worker-health` command the container probes."""

import asyncio
import os
import time

import pytest
from click.testing import CliRunner

from apps.worker.heartbeat import Heartbeat, heartbeat_age
from personalos import cli
from personalos.config import settings


async def test_the_heartbeat_beats_while_running_and_is_removed_on_stop(tmp_path):
    path = tmp_path / "beat"
    heartbeat = Heartbeat(path, interval=0.01)
    stop = asyncio.Event()

    task = asyncio.create_task(heartbeat.run(stop))
    for _ in range(100):
        if path.exists():
            break
        await asyncio.sleep(0.01)
    assert path.exists()
    assert heartbeat_age(path) < 5

    stop.set()
    await asyncio.wait_for(task, timeout=2)
    assert not path.exists()


def test_no_heartbeat_has_no_age(tmp_path):
    assert heartbeat_age(tmp_path / "absent") is None


def _cli(monkeypatch, path, *args):
    monkeypatch.setattr(settings, "worker_heartbeat_file", str(path) if path else "")
    monkeypatch.setattr(settings, "worker_heartbeat_seconds", 10.0)
    return CliRunner().invoke(cli.cli, ["worker-health", *args])


def test_worker_health_passes_on_a_fresh_heartbeat(monkeypatch, tmp_path):
    Heartbeat(tmp_path / "beat").beat()

    result = _cli(monkeypatch, tmp_path / "beat")

    assert result.exit_code == 0, result.output
    assert "worker alive" in result.output


def test_worker_health_fails_on_a_stale_heartbeat(monkeypatch, tmp_path):
    path = tmp_path / "beat"
    Heartbeat(path).beat()
    old = time.time() - 120
    os.utime(path, (old, old))

    result = _cli(monkeypatch, path)

    assert result.exit_code != 0
    assert "old" in result.output


@pytest.mark.parametrize("configured", [True, False])
def test_worker_health_fails_without_a_heartbeat(monkeypatch, tmp_path, configured):
    result = _cli(monkeypatch, tmp_path / "beat" if configured else None)

    assert result.exit_code != 0


async def test_the_worker_runs_its_heartbeat_beside_its_loops(tmp_path):
    from apps.worker.main import Worker

    class Queue:
        pass

    path = tmp_path / "beat"
    worker = Worker(commands=Queue(), runners={}, heartbeat=Heartbeat(path, interval=0.01))
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run(stop))
    for _ in range(100):
        if path.exists():
            break
        await asyncio.sleep(0.01)
    assert path.exists()

    stop.set()
    await asyncio.wait_for(task, timeout=2)
