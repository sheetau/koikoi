V4 work calibration diagnosis
============================

This is a descriptive decomposition of the completed V4 counters, not a new
evaluation or a causal experiment. It does not change V4's failed full
matched-work criterion. The input is
``docs/paper/generated/public_stock_v4_evidence.json`` (SHA256
``694b550472811cf3f1bb5fc1511eec01f5771420795ebd132d4b4d9d493e089e``).

For a fixed number p of worlds per legal action, the following identity
separates physical work into measured quantities::

  transitions / (p * matches)
    = (roots / matches)
      * (rollouts / (p * roots))
      * (transitions / rollouts)

The middle factor is mean legal actions per searched root; the last is mean
physical transitions per completed-round rollout. All three factors come
from recorded planning counters, not independent rollout replay.

For the failed all-three parent-0 high arm, the pilot versus held-out values
are: 5.890625 versus 5.3828125 rounds per match; 80.90625 versus 72.890625
searched roots per match; 3.470452 versus 3.472294 legal actions per root;
and 17.300106 versus 17.278304 transitions per rollout. Roots per match fall
9.9073%, whereas the latter two factors change by +0.0531% and -0.1260%.
Thus the observed per-world work difference is almost entirely associated
with fewer searched decisions per match, not shorter individual simulations.

Per-world work changes from 4857.545410 to 4373.098135 transitions per match,
a ratio of 0.900269. Multiplying the latter by the frozen 49 worlds yields
214281.808594, or -10.715913% relative to the 240000 target. The pilot uses
32 duplicate streams and 32 worlds; evaluation uses 128 different streams
and 49 worlds. Both deal composition and search-induced trajectories can
change match length, so these aggregates cannot identify their separate
causal contributions. In particular, they do not prove that stronger search
itself shortened the matches.

The held-out/pilot per-world work ratios for the six high arms are:

================  ========  ========
Rule              Parent 0  Parent 1
================  ========  ========
Control           1.023456  1.026434
Preview + choice  0.990190  0.953136
All three         0.900269  0.940228
================  ========  ========

Implication for a new design
---------------------------

One small pilot at a single particle count is insufficient to ensure a fixed
expected-work tolerance after changing budgets. A further comparison should
measure work at its proposed low and high counts on separate calibration
streams, apply a prespecified work-only correction, and freeze all counts
before a new evaluation cohort. More calibration streams reduce sampling
variation but cannot guarantee equality: stopping trajectories remain
endogenous to the search policy. The held-out tolerance must still be checked
and failures reported; no tolerance change or post-result retuning is allowed.

A hard transition allowance would answer a different question: performance
under equal maximum resources rather than equal realized work. It would also
require a new scheduler and validation of completed-round accounting. It must
not be introduced merely by renaming the existing expected-work comparison.

Additional adaptation must first pass V5A's independent qualification. This
diagnosis does not select a rule, checkpoint or stronger-policy search result,
authorize reuse of an old policy's table for a new policy, or consume any new
calibration/evaluation streams.
