"""Mechanical gate-scope repair; retain the original 4,096 ordered source cases."""

from scripts.frontier_incremental_design import cases, summarize  # noqa: F401
from scripts.frontier_incremental_design import specification as original


def specification():
    result = original()
    result.update(
        version="independent-uniform-frontier-increment-scope-v2",
        boundary="five draws: next learner discard; fewer draws: unchanged old controller",
        amendment="Correct the <=5 source gate dispatch; retain all 15 previously acquired sources at their original case indices; generate only missing cases",
        extension="none; retain all source cases, including those preceding the mechanical scope correction",
    )
    return result
