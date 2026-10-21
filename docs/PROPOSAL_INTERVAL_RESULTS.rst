Six-draw contingent proposal screen: complete
===========================================

All eight fixed fixtures passed. Three complete candidates were independently
verified at values 1, 0 and 1; none improved its parent. Five proposal searches
exhausted their one-million-transition caps without completing a candidate.
No policy was adopted and every actual trace/outcome remained unchanged.
There is no additional playing-strength result from this screen.

.. csv-table:: Fixed proposal/reference allowance, no borrowing
   :header: Source,Complete candidate value,Proposal decisions,Reference decisions,Result

   1,1,74,74,no strict gain
   19,0,149700,0,no strict gain
   88,none,999980,0,proposal cap
   112,none,1000000,0,proposal cap
   237,1,44,44,no strict gain
   251,none,999992,0,proposal cap
   322,none,999987,0,proposal cap
   738,none,999996,0,proposal cap

Every completed candidate, including non-adoptions, was fully replayed through
independent Python transitions and all remaining draw branches. The saved
programs contain 2, 2,121 and 1 public overrides respectively. No native policy
cache or symbolic outcome replaces the oracle's terminal result. A candidate
of value zero can be rejected without completing a parent whose value is at
least zero. The two other parent references are exact at one.

Work and checks
---------------

Proposal work is 5,149,773 physical decisions / 3,492,677 completions and
976,966,281 represented decisions / 248,286,312 represented completions.
Reference work is 118 physical decisions / 81 completions and 255,134,880
represented decisions / 175,134,960 represented completions. Total newly
executed planning decisions are 5,149,891. Lower work than earlier screens
partly reflects earlier refusal to produce candidates; it is not an equivalent-
solution efficiency result. Operational concurrent timings are retained in
the review and do not establish isolated latency speedup.

Independent candidate oracles use 75,372 physical decisions / 54,123 terminal
completions and 271,277,664 represented decisions / 179,459,280 completions.
Support reconstruction uses 60,508 replay decisions, including 31,954 candidate
and 28,554 opponent decisions / teacher calls. Actual execution and replay
each use 42 decisions across 16 terminal tails. No new source or chance draw
is introduced. These categories are not combined into a match count.

The complete-only review rehashes all bindings and programs, recompiles each
candidate from its complete same-belief proposal, reconstructs the selection,
checks separate caps and resolved reference mass, and validates every required
oracle and actual-tail receipt. Incomplete export was refused; complete export
passed. Ruff passes. The previously verified core/bridge tests remain unchanged.

Protocol SHA256:
``20a31067a191cfd96dcef1c75ef1b20184d50c143564b43cf635f34b6cf5f439``.
Summary SHA256:
``d6e83d7f6f8e66ddbaea51a145d0aa7aa595d155c52db2482ec733ccb7dc85f7``.
Complete review: ``runs/proposal_interval_screen_review.json``.
SHA256: ``b11a44a9566bc23e572bf757584d960c79b445ad67bf7566113af3c3c88abd13``.
Session9032 and queue68556 exited successfully; this screen is closed.

Next development question
-------------------------

The six-draw screen combines earlier entry with a difficult proposal tree.
The next fixed comparison retains the original five-draw gate and targets the
original controller's unvalued-parent fallback. Select the first such source
per parent/opponent/seat before new planning: 16, 168, 246, 387, 393, 433, 867,
1054. Keep the same added 1M proposal + 1M reference allowance. Import the
original failed fixed-phase cost separately so the nominal online total stays
within the original four-million two-phase maximum. This is an isolated
extension feasibility screen, not an already deployed hybrid controller.

Existing successful selections must be preserved in any later integration.
The targeted reused fixtures cannot supply a new population confidence claim;
fresh prospective evaluation remains necessary. See
``FIVE_DRAW_FALLBACK_SCREEN.rst`` for the frozen selection and accounting rules.
