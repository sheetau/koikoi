Delayed public identification: completed diagnostic
==================================================

All 2,048 retained training prefixes were processed, without reading held-out
evaluation trajectories. They share 512 distinct deals across parents/seats.
The finite candidate set contains the pinned deterministic, assisted CPU
RL-WP and RL-Point policies. Labels are used only to score results.

================  ==========  ==========  ==========  =====================
Prefix group      Correct     Ambiguous  Incorrect   Correct at final root
================  ==========  ==========  ==========  =====================
Parent 1 / WP     428 / 512   84          0           410 / 483
Parent 1 / Point  435 / 512   77          0           387 / 455
Parent 2 / WP     438 / 512   74          0           402 / 460
Parent 2 / Point  445 / 512   67          0           399 / 458
================  ==========  ==========  ==========  =====================

There are 1,746 correct identifications and 302 unresolved prefixes. Neither
an empty candidate set nor exclusion of the true model occurs. At natural
final-round roots, 1,598 of 1,856 cases have an identified opponent (86.10%).
The tracker makes 44,170 strict CPU predictions: 20,553 WP and 23,617 Point.
All 36,977 reconstructed historical inputs and assistance masks match the
original upstream oracle. The tracker receives only redacted public snapshots
and already observed actions; the oracle is a separate verification path.

Independent replay verifies revelation timing, exclusions, first identification,
source endpoints and all group counts, without calling the tracker or repeating
neural inference. Replay accounts for 212,212 decisions, 14,101 round endings
and 192 early full-match completions; it adds no independent deals. The other
1,856 trajectories stop at the final-round start and are incomplete prefixes.

Evidence
--------

* Producer: ``runs/revealed_opponent_identification/audit.json``, SHA256
  ``d26b04a79262d35fd74d9a9c8e3497d8f7927ea919b3f67db90686dde078e2e2``.
* Independent audit: same directory, ``independent_audit.json``, SHA256
  ``c83ea431e670ef91b3fb53d780993e6bcc3ebd92a1df54387ae627f583096fc2``.
  All 29 input identities reverified after completion.
* Producer queue 81291 and independent audit both exit zero.

Interpretation and next intervention
-----------------------------------

This establishes a usable public opponent signal within the declared two-model
class. It does not establish strength, human-opponent identification or reliable
identification of unlisted/stochastic policies. Opponent modelling and policy
conditioning are established ideas: Bayes' Bluff (Southey et al., UAI 2005),
Bayesian Policy Reuse (Rosman et al., 2016), and DRON (He et al., ICML 2016;
https://proceedings.mlr.press/v48/he16.html) are relevant precedents.

The next isolated intervention conditions final-round adaptation on the public
candidate set, with a matched masked-memory control. Both use the same natural
roots and unchanged rounds 1--7 parent. Unknown or inconsistent support maps to
zero bits; no ground-truth opponent label may fill missing predictions. A new
prospective comparison must establish any playing-strength claim.
