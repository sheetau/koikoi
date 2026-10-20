import pytest

from scripts import finish_score_auxiliary as queue


class FakeProducer:
    def __init__(self, codes):
        self.codes = iter(codes)

    def exit_code(self):
        return next(self.codes)


def test_early_completion_receipt_does_not_release_live_process(tmp_path, monkeypatch):
    queue.write_new(
        tmp_path / "complete.json",
        {
            "status": "complete",
            "protocol_sha256": queue.TRAINING_SHA,
        },
    )
    sleeps = []
    monkeypatch.setattr(queue.time, "sleep", sleeps.append)
    queue.wait_for_training(FakeProducer([None, 0]), tmp_path, poll=3)
    assert sleeps == [3]


def test_failed_process_exit_overrides_receipt(tmp_path):
    queue.write_new(
        tmp_path / "complete.json",
        {
            "status": "complete",
            "protocol_sha256": queue.TRAINING_SHA,
        },
    )
    with pytest.raises(RuntimeError, match="exited 9"):
        queue.wait_for_training(FakeProducer([9]), tmp_path)


def test_zero_exit_needs_correct_completion(tmp_path):
    with pytest.raises(FileNotFoundError):
        queue.wait_for_training(FakeProducer([0]), tmp_path)
    queue.write_new(
        tmp_path / "complete.json", {"status": "complete", "protocol_sha256": "wrong"}
    )
    with pytest.raises(RuntimeError, match="matching completion"):
        queue.wait_for_training(FakeProducer([0]), tmp_path)


def test_exclusive_queue_and_training_failure(tmp_path):
    with pytest.raises(FileExistsError):
        queue.finish(tmp_path, FakeProducer([]))
    queue.write_new(tmp_path / "failed.json", {"error": "fixture"})
    with pytest.raises(RuntimeError, match="failure receipt"):
        queue.wait_for_training(FakeProducer([]), tmp_path)


def test_audit_precedes_fixed_main_evaluation():
    stages = queue.jobs()
    assert [name for name, _ in stages] == [
        "training_audit",
        "prepare",
        "evaluate",
        "evaluation_audit",
    ]
    assert stages[0][1][0] == "scripts.audit_score_auxiliary_training"
    assert all("capability" not in arguments for _, arguments in stages)
    assert all("101000000" not in str(arguments) for _, arguments in stages)
