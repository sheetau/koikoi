Opening-only card search
========================

Reason for this intervention
----------------------------

The fixed late-koi comparison found no action changes for either estimator.
All twelve independent reference mean differences favored the retained parent's
choice. This does not establish global optimality,but it does not motivate more
particles at those same selected roots. Investigate earlier card choices.

The existing history-search cap prevents exploration when the hand space has
more than100 candidates. At the opening dealer discard,there has not yet been
an opponent action. No action-likelihood enumeration is needed at that particular
root,even though its hand space is large.

Uniform-deal argument and limitations
------------------------------------

The dealer knows eight own cards and eight field cards. Of the32 unknown cards,
eight belong to the opponent and24 are stock. There are10,518,300 candidate hand
subsets before initial-deal validity rejection. Under an ideal uniformly shuffled
deal,fix any ordering of the known cards:each valid other-hand subset then has
the same8!*24! remaining-slot multiplicity. Summing over the known cards'
unobserved slot order adds a common factor. The known own
hand/field are already valid. Rejecting other hands with four cards of one month
therefore samples uniformly among the remaining valid subsets. Conditional on
that hand,shuffle the remaining stock uniformly. Hand slot order is irrelevant
to the simulator's actions and published observations.

No opponent action has occurred,so its likelihood is one for every hand/model
hypothesis. Keep the declared equal WP/Point model prior;do not pass the actual
opponent identity. The existing sample_hidden_batch implementation already
performs the required unweighted choice/rejection/stock shuffle and replaces
future seeds. The opening adapter changes eligibility,not this sampling code.

This is a standard conditional-sampling argument,not a new Bayesian method.
It is exact for the stated ideal shuffle model. It does not enumerate the native
finite PRNG's seed space conditioned on cards/dealer,or establish calibrated
likelihoods for arbitrary human opponents. Initial dealer selection is unchanged.

Restricted policy
-----------------

scripts/opening_search.py provides OpeningTerminalSearch and OpeningRoundSearch.
Both search only round index0,turn0,discard phase,with the acting player equal
to the dealer. Public counts must show eight legal own cards and32 unknown cards.
All seven alternatives to the strict parent action share the same sampled worlds
and opponent labels. Candidate selection and fresh confirmation use the existing
validated comparison implementations. Round search additionally separates A/B
tables and includes their joint uncertainty.

Every other state uses the original parent/public endgame guard without search.
In particular,the nondealer's first turn already contains an opponent action and
is not treated as an empty-history root. Later-round starts are also excluded.
Only the candidate's dealer games can change at this opening. After that one
choice,both rollout and actual proposed deployment continue with the parent.
Thus a future match study can isolate a single card decision,without an unknown
effect from repeated future search. This restriction is not a strength claim.

Validation and next evidence
----------------------------

Five CPU/CUDA tests pass. For both dealer seats,128 sampled worlds retain exactly
the root observation and satisfy the initial-deal rule;actual hidden reassignment
does not alter generated worlds. Both estimators compare all eight cards,use
fresh confirmation,leave the root unchanged,and return to the exact parent
outside the opening without additional simulation. The tests force acceptance
only to exercise the confirmation path. Synthetic boundary tables are not used
as measured policies.

No real-table opening diagnostic or strength evaluation has run. Freeze a new
comparison's roots,all-action particle budgets,independent terminal-reference
targets and stopping rule before execution. Retain all alternatives and missing
roots;do not extend the completed late-koi study or select the better A/B table.
The canonical playable model and manuscript remain unchanged.
