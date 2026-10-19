import pytest

from scripts import finish_point_initialization as queue


class FakeProducer:
    def __init__(self, codes):
        self.codes = iter(codes)

    def exit_code(self):
        return next(self.codes)


def test_early_receipt_cannot_release_live_producer(tmp_path, monkeypatch):
    queue.write_new(
        tmp_path / "complete.json",
        {"status": "complete", "protocol_sha256": queue.TRAINING_SHA},
    )
    waits = []
    monkeypatch.setattr(queue.time, "sleep", waits.append)
    queue.wait_for_training(FakeProducer([None, 0]), tmp_path, poll=3)
    assert waits == [3]


def test_failed_exit_overrides_completion(tmp_path):
    queue.write_new(
        tmp_path / "complete.json",
        {"status": "complete", "protocol_sha256": queue.TRAINING_SHA},
    )
    with pytest.raises(RuntimeError, match="exited 9"):
        queue.wait_for_training(FakeProducer([9]), tmp_path)


def test_clean_exit_requires_matching_completion(tmp_path):
    with pytest.raises(FileNotFoundError):
        queue.wait_for_training(FakeProducer([0]), tmp_path)
    queue.write_new(
        tmp_path / "complete.json", {"status": "complete", "protocol_sha256": "wrong"}
    )
    with pytest.raises(RuntimeError, match="matching completion"):
        queue.wait_for_training(FakeProducer([0]), tmp_path)


def test_queue_root_is_exclusive_and_failure_receipt_blocks_wait(tmp_path):
    with pytest.raises(FileExistsError):
        queue.finish(tmp_path, FakeProducer([]))
    queue.write_new(tmp_path / "failed.json", {"error": "failed"})
    with pytest.raises(RuntimeError, match="failure receipt"):
        queue.wait_for_training(FakeProducer([]), tmp_path)
