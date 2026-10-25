"""Actual evaluator parser check for the pre-evaluation recovery command."""

import subprocess
import sys

from scripts.complete_public_stock_adaptation import evaluation_command
from scripts.public_stock_adaptation import EVALUATE


def test_recovery_command_is_accepted_by_actual_evaluator_parser(tmp_path):
    out = tmp_path / "must_not_be_created"
    command = evaluation_command(
        {**EVALUATE, "checkpoint": "unused.pt", "opponent": "other.pt", "out": str(out)}
    )
    assert command[:2] == ["scripts.evaluate_public_stock_residual", "evaluate"]
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", *command, "--help"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert "--checkpoint" in result.stdout and "--pairs" in result.stdout
    assert not out.exists()
