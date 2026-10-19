Koi-temperature comparison: complete development results
========================================================

The prespecified replicated improvement criterion fails.
It requires positive lower win-change bounds against both the matched
fixed-temperature control and the retained parent at both training seeds.
The gate uses unrounded values; displayed values are rounded to four decimals.

All six policies share 4,096 development deal streams: 2,048 per opponent,
with both seats played. The study contains 49,152 full matches. Draws count
as half wins. The mixture gives equal weight to assisted WP and Point.
Parent 1/2 means retained A/B; new learner seeds are 96110001/96210001.
Both new conditions start from original WP, not the retained parents.

Primary comparisons
-------------------

====  ===================  ==========================  =========
Seed  Comparison           Win change (pp), 95% CI     Lower > 0
====  ===================  ==========================  =========
1     calibrated - fixed   -0.1953 [-1.1108, +0.7446]  no
1     calibrated - parent  +0.0122 [-0.8972, +0.9094]  no
2     calibrated - fixed   -0.7751 [-1.7274, +0.1770]  no
2     calibrated - parent  -0.7202 [-1.5991, +0.1588]  no
====  ===================  ==========================  =========

All paired comparisons
----------------------

====  =======  ===================  ==========================  ==========================
Seed  Scope    Comparison           Win change (pp), 95% CI     Point change, 95% CI
====  =======  ===================  ==========================  ==========================
1     mixture  calibrated - fixed   -0.1953 [-1.1108, +0.7446]  -0.9392 [-1.4751, -0.3933]
1     mixture  calibrated - parent  +0.0122 [-0.8972, +0.9094]  -0.4319 [-0.9634, +0.0938]
1     mixture  fixed - parent       +0.2075 [-0.7141, +1.0986]  +0.5073 [-0.0203, +1.0085]
1     wp       calibrated - fixed   -0.4883 [-1.7822, +0.8179]  -1.1792 [-1.9151, -0.4185]
1     wp       calibrated - parent  +0.2319 [-1.0986, +1.5625]  -0.2881 [-1.0449, +0.4761]
1     wp       fixed - parent       +0.7202 [-0.5740, +2.0511]  +0.8911 [+0.1650, +1.6011]
1     point    calibrated - fixed   +0.0977 [-1.1722, +1.4038]  -0.6992 [-1.4551, +0.0679]
1     point    calibrated - parent  -0.2075 [-1.4648, +1.0376]  -0.5757 [-1.3418, +0.1641]
1     point    fixed - parent       -0.3052 [-1.5625, +0.9399]  +0.1235 [-0.6377, +0.8375]
2     mixture  calibrated - fixed   -0.7751 [-1.7274, +0.1770]  -0.0596 [-0.5982, +0.4688]
2     mixture  calibrated - parent  -0.7202 [-1.5991, +0.1588]  -0.3242 [-0.8232, +0.1785]
2     mixture  fixed - parent       +0.0549 [-0.8606, +1.0071]  -0.2646 [-0.7896, +0.2493]
2     wp       calibrated - fixed   -0.3540 [-1.6971, +1.0135]  +0.3403 [-0.4033, +1.0854]
2     wp       calibrated - parent  -0.6348 [-1.9287, +0.5984]  -0.5732 [-1.2910, +0.1519]
2     wp       fixed - parent       -0.2808 [-1.6479, +1.0864]  -0.9136 [-1.6841, -0.1465]
2     point    calibrated - fixed   -1.1963 [-2.4780, +0.1099]  -0.4595 [-1.2041, +0.2837]
2     point    calibrated - parent  -0.8057 [-2.0020, +0.4150]  -0.0752 [-0.7886, +0.6451]
2     point    fixed - parent       +0.3906 [-0.8667, +1.6724]  +0.3843 [-0.3403, +1.0957]
====  =======  ===================  ==========================  ==========================

All absolute results
--------------------

====  =======  ==========  ==========================  ==========================
Seed  Scope    Policy      Win score (%), 95% CI       Point margin, 95% CI
====  =======  ==========  ==========================  ==========================
1     mixture  parent      51.3611 [50.5554, 52.1790]  +0.3950 [-0.0876, +0.8879]
1     mixture  fixed       51.5686 [50.7079, 52.4353]  +0.9023 [+0.3721, +1.4229]
1     mixture  calibrated  51.3733 [50.5371, 52.2461]  -0.0369 [-0.5474, +0.4808]
1     wp       parent      51.6724 [50.5615, 52.8198]  +2.2520 [+1.6025, +2.9234]
1     wp       fixed       52.3926 [51.2573, 53.5522]  +3.1431 [+2.4609, +3.8169]
1     wp       calibrated  51.9043 [50.7812, 53.0518]  +1.9639 [+1.2939, +2.6309]
1     point    parent      51.0498 [49.8535, 52.2705]  -1.4619 [-2.2251, -0.6899]
1     point    fixed       50.7446 [49.5361, 51.9775]  -1.3384 [-2.1207, -0.5562]
1     point    calibrated  50.8423 [49.5972, 52.1118]  -2.0376 [-2.7896, -1.2563]
2     mixture  parent      51.3489 [50.5005, 52.1790]  +0.3958 [-0.1201, +0.9004]
2     mixture  fixed       51.4038 [50.5676, 52.2705]  +0.1311 [-0.3772, +0.6367]
2     mixture  calibrated  50.6287 [49.7620, 51.4893]  +0.0715 [-0.4280, +0.5833]
2     wp       parent      51.8066 [50.6836, 52.9541]  +2.7139 [+2.0361, +3.3960]
2     wp       fixed       51.5259 [50.3784, 52.6978]  +1.8003 [+1.1440, +2.4761]
2     wp       calibrated  51.1719 [50.0366, 52.3438]  +2.1406 [+1.4785, +2.8091]
2     point    parent      50.8911 [49.6826, 52.0996]  -1.9224 [-2.7061, -1.1382]
2     point    fixed       51.2817 [50.0610, 52.5269]  -1.5381 [-2.3022, -0.7686]
2     point    calibrated  50.0854 [48.8522, 51.2939]  -1.9976 [-2.7725, -1.2358]
====  =======  ==========  ==========================  ==========================

Interpretation and work
-----------------------

Intervals use 10,000 paired bootstrap replicates (seed 97310001). Each
block retains both seats and both fixed opponent strata. These are marginal
95% intervals, not simultaneous intervals for all displayed comparisons.
They measure deal variation, not the variation of new training seeds.
Point margins are candidate minus opponent; divide by two for the original
paper's final-minus-initial point convention. CSV files retain unrounded
values and use fractions for wins and win changes, not percent units.

Training work (warmup included): {"decisions": 34078720, "matches": 293524, "rounds": 2285698, "updates": 2080}
Actual evaluation work: {"decisions": 5706119, "full_matches": 49152, "round_endings": 383234}
Recorded-action replay work: {"decisions": 5706119, "full_matches": 49152, "round_endings": 383234}

The independent audit checks saved training states, recorded trajectories
and paired statistics. It does not rerun training or neural action selection.
This export adds no neural calls, simulator steps, matches or training.
A positive development result needs prospective confirmation before promotion.
No model, checkpoint or temperature is selected by this report.

Mechanism and next decision
---------------------------

All 18 paired win-change intervals include zero. Four secondary point-change
intervals exclude zero, with mixed signs across the fixed controls; they do
not establish the primary win improvement. Every evaluated policy has a
negative point-margin interval against Point on these development deals.
The original confirmed results and the canonical human-play model remain
unchanged. Reserved 101M confirmation deals are not used for this failed gate.

The separate common-input diagnostic found higher koi entropy at the declared
calibrated temperatures and wider learned raw score gaps. This comparison
shows that those descriptive changes did not establish a stronger player at
the fixed training budget. It does not prove that all temperatures are
ineffective or that the policies are equivalent. Do not extend this study,
select an intermediate checkpoint or retune its temperature after these results.
Further methods require a separate design and fresh development evaluation.

The contribution of this completed experiment is the controlled test and
its score/probability diagnosis. Temperature calibration itself is established
methodology; KOI_TEMPERATURE_PRIOR_ART.rst records the directly relevant
policy-distillation and discrete entropy-control literature. This result does
not satisfy the active goal of further replicated playing-strength improvement.

Evidence identities
-------------------

Artifacts reside under runs/koi_temperature_study and
runs/koi_temperature_results. The finalizer session64818 exits0; its worker
36936 and the producer/evaluation workers are gone. All completed outputs are
preserved; do not restart either study or export into these directories.

* Protocol: b9075ac261ce6a3e681a64fa7845f031850ef20fba60f618435b3ad108a74d19,91 inputs.
* Independent audit: c1c89bf669f48490b51a40e6cebb35cf88a115b232fa95c6fe9f0ae1be625b45,169 inputs.
* Summary: 247a230efbb275f58658be6f89c6a789fa9cc9c4b4c26ad2de43673c21b9e26a,171 inputs.
* Export evidence: e24777f4ac07c4cf32955a170902aea789c56ae960d43ea609c3fdbb57cb7877,174 inputs.
* Generated RST: 75c061c367e0609e9f7a35d325c4a7bf0c9b3800cf8e64b93c513168dfc1f1c9.
* Absolute CSV: 58db4ea686aef3af12bb0861b317fddfdb405087a0aa3dc9dd274c9780935629.
* Contrast CSV: 265e81332d0991195781085ddfe8f88af587c486fb32a7c16cbabcc07f9a366b.

All CSV identities and numeric values were separately compared to the
completed audit-derived summary. The output hashes agree. This presentation
check is not a new bootstrap, trajectory replay or neural evaluation.
