Duration comparison: complete development results
=================================================

The prespecified four-contrast improvement criterion fails.
All four unrounded simultaneous lower bounds must exceed zero.
A passing development gate needs separate prospective confirmation.

Group parent1: fresh seed 99110001, retained comparator A.
Group parent2: fresh seed 99210001, retained comparator B.
Short and long start from original WP and share one training trajectory.
Short: 500 policy updates; long: 4,000. Each includes 20 warmup updates.
This is unequal-budget scaling, not a new algorithm or equal-compute gain.

98,304 matches use 8,192 shared initial deals and both seats.
Draws count as half wins; mixture weights assisted WP and Point equally.
4,096 compound blocks pair WP/Point deals and retain all policies/seats.
10,000 bootstrap replicates use seed 99410001. Primary intervals form one
simultaneous 95% family; secondary intervals are descriptive marginal 95%.
They measure deal variation, not variation across new training seeds.

Primary comparisons
-------------------

===============================  =================================  =========
Comparison                       Win change (pp), simultaneous 95%  Lower > 0
===============================  =================================  =========
parent1/mixture/long-parent/win  +0.5249 [-0.3906, +1.4404]         no
parent1/mixture/long-short/win   +0.5798 [-0.3191, +1.4788]         no
parent2/mixture/long-parent/win  +0.6104 [-0.2847, +1.5054]         no
parent2/mixture/long-short/win   +0.8942 [+0.0005, +1.7878]         yes
===============================  =================================  =========

All absolute results
--------------------

=====  =======  =================  ==========================  ==========================
Group  Scope    Policy/comparison  Win score (%)               Point margin
=====  =======  =================  ==========================  ==========================
1      mixture  parent             51.3123 [50.7294, 51.9349]  +0.4386 [+0.0741, +0.8189]
1      mixture  short              51.2573 [50.6653, 51.8585]  +0.0227 [-0.3474, +0.3916]
1      mixture  long               51.8372 [51.2421, 52.4353]  +1.2233 [+0.8456, +1.5972]
1      wp       parent             52.1912 [51.3794, 53.0090]  +2.9622 [+2.4826, +3.4278]
1      wp       short              51.5686 [50.7507, 52.3865]  +2.0706 [+1.5815, +2.5515]
1      wp       long               52.7283 [51.8982, 53.5828]  +3.6338 [+3.1267, +4.1406]
1      point    parent             50.4333 [49.5850, 51.3124]  -2.0850 [-2.6394, -1.5098]
1      point    short              50.9460 [50.0671, 51.8372]  -2.0251 [-2.5891, -1.4622]
1      point    long               50.9460 [50.0916, 51.8127]  -1.1873 [-1.7510, -0.6294]
2      mixture  parent             51.4038 [50.8118, 51.9958]  +0.3563 [-0.0110, +0.7255]
2      mixture  short              51.1200 [50.5280, 51.7090]  +0.3927 [+0.0256, +0.7643]
2      mixture  long               52.0142 [51.4130, 52.6124]  +1.3112 [+0.9369, +1.6846]
2      wp       parent             52.0447 [51.2634, 52.8442]  +2.7771 [+2.3054, +3.2491]
2      wp       short              51.8799 [51.0986, 52.6794]  +2.9055 [+2.4363, +3.3733]
2      wp       long               51.8066 [50.9644, 52.6367]  +2.9602 [+2.4473, +3.4666]
2      point    parent             50.7629 [49.9084, 51.6357]  -2.0645 [-2.6177, -1.4988]
2      point    short              50.3601 [49.4812, 51.2514]  -2.1201 [-2.6768, -1.5532]
2      point    long               52.2217 [51.3489, 53.0945]  -0.3379 [-0.8889, +0.2339]
=====  =======  =================  ==========================  ==========================

All paired comparisons
----------------------

=====  =======  =================  ==========================  ==========================
Group  Scope    Policy/comparison  Win change (pp)             Point change
=====  =======  =================  ==========================  ==========================
1      mixture  long-short         +0.5798 [-0.1374, +1.2939]  +1.2006 [+0.7795, +1.6196]
1      mixture  long-parent        +0.5249 [-0.2045, +1.2573]  +0.7847 [+0.3480, +1.2129]
1      mixture  short-parent       -0.0549 [-0.7294, +0.6042]  -0.4159 [-0.7940, -0.0422]
1      wp       long-short         +1.1597 [+0.1282, +2.2034]  +1.5632 [+0.9629, +2.1565]
1      wp       long-parent        +0.5371 [-0.5188, +1.5810]  +0.6716 [+0.0769, +1.2688]
1      wp       short-parent       -0.6226 [-1.5993, +0.3357]  -0.8916 [-1.4204, -0.3625]
1      point    long-short         +0.0000 [-0.9949, +0.9889]  +0.8379 [+0.2319, +1.4294]
1      point    long-parent        +0.5127 [-0.4883, +1.5198]  +0.8977 [+0.2871, +1.4898]
1      point    short-parent       +0.5127 [-0.3723, +1.3977]  +0.0598 [-0.4653, +0.5886]
2      mixture  long-short         +0.8942 [+0.1709, +1.6052]  +0.9185 [+0.5010, +1.3373]
2      mixture  long-parent        +0.6104 [-0.1129, +1.3275]  +0.9548 [+0.5186, +1.3823]
2      mixture  short-parent       -0.2838 [-0.8973, +0.3448]  +0.0364 [-0.3234, +0.4021]
2      wp       long-short         -0.0732 [-1.0742, +0.9338]  +0.0547 [-0.5510, +0.6377]
2      wp       long-parent        -0.2380 [-1.2512, +0.7997]  +0.1831 [-0.4238, +0.7783]
2      wp       short-parent       -0.1648 [-1.0620, +0.7446]  +0.1284 [-0.3740, +0.6384]
2      point    long-short         +1.8616 [+0.8545, +2.8503]  +1.7822 [+1.1760, +2.3755]
2      point    long-parent        +1.4587 [+0.4332, +2.4353]  +1.7266 [+1.1028, +2.3259]
2      point    short-parent       -0.4028 [-1.2880, +0.4700]  -0.0557 [-0.5750, +0.4617]
=====  =======  =================  ==========================  ==========================

Work and interpretation
-----------------------

Training totals (short prefixes included once): {"decisions": 131727360, "matches": 1133134, "rounds": 8804257, "updates": 8040}
Actual evaluation and separately replayed work: {"decisions": 11376824, "full_matches": 98304, "round_endings": 762598}
Point margins are candidate minus opponent; divide by two for the
original paper's final-minus-initial point convention.
The audit replays recorded actions and checks statistics; it does not
rerun training or neural inference. This export adds no matches.
JSON preserves unrounded values in fraction units for win scores.
No endpoint is selected or promoted by this report.

Administrative recovery
-----------------------

The original evaluation stopped on a Windows progress-file replacement error.
Eight completed cells were copied unchanged; four incomplete cells were
rerun on the identical full schedule. All preserved complete prefix rows
match repeated neural inference exactly. The original failed receipts remain.
The 98,304 analysis matches and replay work above exclude the additional
recorded duplicate execution below; these repeats add no independent deals.
Additional recorded duplicate work: {"decisions": 667309, "full_matches": 5753, "round_endings": 44637}
Unrecorded interrupted work: unknown.
Separate recovery capability execution and native replay: {"decisions": 918, "full_matches": 8, "round_endings": 64}
Capability matches repeat existing fixtures and are not strength evidence.

Interpretation and next decision
-------------------------------

Both long endpoints have descriptive marginal win intervals above 50% against
both published opponents. Against Point, wins are 50.9460% [50.0916,51.8127]
and52.2217% [51.3489,53.0945]. These opponent-specific secondary intervals are
not a simultaneous superiority claim or a prospective final confirmation.
The first seed's short and long Point win estimates are exactly equal; the
stronger second-seed result must not be generalized to both learning runs.

Both long endpoints improve mixture point margins relative to both short and
retained comparators, with all four descriptive paired intervals above zero.
Against Point their mean margins remain negative: -1.1873 and-0.3379; the
second interval includes zero. Thus improved expected points do not establish
positive expected points against Point. The four primary simultaneous win
intervals permit only one positive comparison (second seed long-minus-short).
The predeclared development qualification fails. No promotion or101Mconfirmation
is authorized by this study, and the current playable policy remains unchanged.

Do not extend training or add deals to turn this closed comparison into a pass.
The result supports measurable score improvements with greater training effort;
it does not establish a replicated win improvement over the retained players,
training convergence, or a new algorithm. The stronger-play/research objective
remains open. Next incorporate this result in the manuscript and specify the
previously identified Point-initialization comparison with separate score-scale
and exploration controls. That comparison addresses an untested practical
initialization choice; a novelty claim would still require a distinct method,
mechanism, prior-art comparison and successful controlled evidence.

Evidence
--------

Passed native-action/statistics audit:
f193d0ac7ce3aa63071f3f65fb81d6b809329944012b4fa48d30c5a9e0367b31.
Summary:74d03c23bb41a2fce755a1aee885119592de4bb527e8ae9f7ce9f870544e0050.
Recovery completion:9640e4891cd7e70c64f3d5c63baf6b9dae5a9a76316743c7fd8663a62c70757b.
Export evidence:runs/published_duration_results/evidence.json,
b4e7bc2566deb3718aaf4c0dd19ac7d270750fe511f933a13e461c0b11c06212,
binding197inputs. All input and exported-output hashes verify. This document
copies the complete generated tables and adds interpretation,not another analysis.
