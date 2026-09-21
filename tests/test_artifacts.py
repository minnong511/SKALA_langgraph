import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

from src.common.artifacts import REDACTED, ArtifactStore
from src.common.events import EventLogger
from src.schemas import AgentResult


def result(attempt=1, **kwargs):
    return AgentResult(
        task_id="technical",
        agent="technical",
        attempt=attempt,
        status="completed",
        summary="검증 완료",
        **kwargs,
    )


def test_attempts_are_immutable_and_runs_cannot_be_reused(tmp_path):
    store = ArtifactStore(tmp_path, "run-001")
    first = store.save_result(result())
    original = first.read_bytes()
    second = store.save_result(result(attempt=2))
    assert first == store.run_dir / "tasks/technical/attempt-1.json"
    assert second != first
    with pytest.raises(FileExistsError):
        store.save_result(result())
    with pytest.raises(FileExistsError):
        ArtifactStore(tmp_path, "run-001")
    assert first.read_bytes() == original
    assert not list(store.run_dir.rglob(".pending-*"))


def test_concurrent_events_are_complete_ordered_and_redacted(tmp_path, capsys):
    logger = EventLogger(tmp_path, "run", secrets=["sk-secret-test"])

    def emit(number):
        # Distinct logger instances must also coordinate writes to the same file.
        target = logger if number % 2 else EventLogger(tmp_path, "run", secrets=["sk-secret-test"])
        return target.emit(
            "search.completed",
            f"result {number}: sk-secret-test",
            task_id=f"task-{number}",
            agent="market",
            duration_ms=number,
            payload={"authorization": "another-secret", "token_count": 12},
        )

    with ThreadPoolExecutor(max_workers=8) as workers:
        returned = list(workers.map(emit, range(64)))
    raw = (tmp_path / "events.jsonl").read_text()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == 64
    assert {row["task_id"] for row in rows} == {row["task_id"] for row in returned}
    assert [row["timestamp"] for row in rows] == sorted(row["timestamp"] for row in rows)
    assert all(datetime.fromisoformat(row["timestamp"]).utcoffset().total_seconds() == 0 for row in rows)
    assert all(row["details"]["payload"] == {"authorization": REDACTED, "token_count": 12} for row in rows)
    assert all(row["details"]["duration_ms"] >= 0 for row in rows)
    terminal = capsys.readouterr().out
    assert len(terminal.splitlines()) == 64
    for secret in ("sk-secret-test", "another-secret"):
        assert secret not in raw + terminal


def test_artifacts_redact_nested_credentials_without_mutation(tmp_path):
    store = ArtifactStore(tmp_path, "redaction", secrets=["shared-private-value"])
    value = {
        "items": [{"API-Key": "hidden-key", "url": "https://example.test/shared-private-value"}],
        "password": "hidden-password",
        "count": 10,
    }
    saved = json.loads(store.save_json("result.json", value).read_text())
    assert saved["items"][0]["API-Key"] == REDACTED
    assert saved["password"] == REDACTED
    assert "shared-private-value" not in saved["items"][0]["url"]
    assert saved["count"] == 10
    assert value["password"] == "hidden-password"
    assert store.save_text("report.md", "key shared-private-value").read_text() == f"key {REDACTED}"


@pytest.mark.parametrize("name", ["../escape.json", "/tmp/escape.json", "a/../../escape.json", "."])
def test_artifact_paths_cannot_escape(tmp_path, name):
    store = ArtifactStore(tmp_path, "paths")
    with pytest.raises(ValueError):
        store.save_json(name, {})


def test_artifact_paths_reject_symlink_escape(tmp_path):
    store = ArtifactStore(tmp_path, "paths")
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.run_dir / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        store.save_text("link/escape.txt", "do not save")
    assert not (outside / "escape.txt").exists()


def test_concurrent_duplicate_writes_publish_one_complete_artifact(tmp_path):
    store = ArtifactStore(tmp_path, "duplicate")

    def save(number):
        try:
            store.save_text("one.txt", str(number) * 10000)
            return number
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=8) as workers:
        outcomes = list(workers.map(save, range(8)))
    winners = [value for value in outcomes if value is not None]
    assert len(winners) == 1
    assert (store.run_dir / "one.txt").read_text() == str(winners[0]) * 10000


def test_invalid_json_never_publishes_a_partial_artifact(tmp_path):
    store = ArtifactStore(tmp_path, "nan")
    with pytest.raises(ValueError):
        store.save_json("broken.json", {"score": float("nan")})
    assert not (store.run_dir / "broken.json").exists()


def test_terminal_reports_measured_duration_retry_reason_and_safe_paths(tmp_path, capsys):
    logger = EventLogger(tmp_path, "terminal", secrets=["private-key"])
    logger.emit(
        "retry_requested",
        "원문 확인 재시도",
        elapsed_seconds=1.25,
        target_agent="market",
        additional_retry=1,
        retry_limit=2,
        reason="응답에서 private-key 오류 발생",
        errors=["private-key rejected"],
        path="tasks/market/attempt-1.json",
        payload={"source_text": "DO-NOT-ECHO-SOURCE"},
    )
    logger.emit(
        "run_end",
        "검토 필요",
        status="needs_review",
        unresolved=["시장 근거 부족"],
        paths={"result": "result.json"},
    )
    output = capsys.readouterr().out
    for expected in (
        "elapsed_seconds=1.25",
        'target_agent="market"',
        "additional_retry=1",
        "retry_limit=2",
        "오류 발생",
        "tasks/market/attempt-1.json",
        "시장 근거 부족",
        "result.json",
    ):
        assert expected in output
    assert "private-key" not in output
    assert "[REDACTED]" in output
    assert "DO-NOT-ECHO-SOURCE" not in output
