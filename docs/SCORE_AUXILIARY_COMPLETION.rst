Score-auxiliary completion queue
================================

scripts/finish_score_auxiliary.py holds a query-only Windows process handle for
the verified live learner. PID reuse cannot substitute another process. It waits
for that process to exit0 and for a matching complete six-arm receipt,then runs:

1. Independent full training/collection/checkpoint audit.
2. Fixed main evaluation preparation,which requires that passed audit.
3. All16main evaluation cells,131,072matches on the predeclared shared deals.
4. Native recorded-action replay and separate statistical verification.

Each stage must exit0 before the next starts. Commands, inputs, process identity,
logs and receipts are preserved. The queue directory and every output are
exclusive. A failure stops progression; no stage is silently restarted. The
existing tested owned-subprocess cleanup is reused from
scripts/finish_point_initialization.py. The queue does not promote a model or
consume101Mconfirmation deals. The fixed mechanism diagnostic is separate.

Command after verifying the actual live learner PID::

    .venv/Scripts/python.exe -m scripts.finish_score_auxiliary --output runs/score_auxiliary_completion --producer-pid 17896

Do not rerun this command if the queue directory already exists. Inspect its
protocol/receipts and held process first. A quiet log while training is live
is expected; it is not a reason to restart learning or the queue.

Five focused tests pass,zero skips,1.53s. They check early receipts while a process
is live,failed exit overriding completion,missing/wrong completion after exit0,
exclusive output and training failure refusal,and audit-before-evaluation order.
These are process-control fixtures with no training or simulated matches.

* Queue source66ae4f83679834cb27c2844517007ab0a6153a927e18f5c969092c82c108c58c.
* Tests9e44a0f9c108adf04872c5c8d504ddaf3e45309f269028f0a7e4c2c041929883.
* XML55cbb2a2be9d3443428e220e0e99324d1d29d6572ef97f3befabf20dadaf2c2d.

Queue completion does not itself establish playing strength. The six simultaneous
primary intervals determine the predeclared development criterion. Result
interpretation,mechanism checks and manuscript integration remain subsequent
work,regardless of which conditions win.
