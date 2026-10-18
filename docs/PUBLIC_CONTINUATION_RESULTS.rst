Multi-round public-memory comparison: results
============================================

The predeclared replicated win-improvement criterion fails. All 18 win-change
intervals include zero. Public memory has positive point effects over masked
learning for parent1 in the balanced mixture and against Point, but the
public-minus-parent win comparisons do not establish a stronger agent.
No model is promoted.

All four 520-update arms and 24,576 strict full matches completed. Policies and
parents share 2,048 fresh 93M deals, with 1,024 paired deals per opponent.
The primary criterion requires positive lower 95% win bounds for both
public-minus-masked and public-minus-parent mixture contrasts in both parents.
The following table retains every primary and secondary contrast. Point units
are final candidate-minus-opponent points.

.. list-table:: Paired changes and 95% intervals
   :header-rows: 1

   * - Parent
     - Contrast
     - Scope
     - Win change (pp)
     - Point change
   * - 1
     - public-minus-masked
     - mixture
     - +0.427 [-0.562, +1.404]
     - +0.629 [+0.116, +1.140]
   * - 1
     - public-minus-masked
     - wp
     - -0.244 [-1.636, +1.123]
     - +0.139 [-0.588, +0.862]
   * - 1
     - public-minus-masked
     - point
     - +1.099 [-0.293, +2.490]
     - +1.119 [+0.395, +1.847]
   * - 1
     - public-minus-parent
     - mixture
     - -0.171 [-1.160, +0.781]
     - -0.065 [-0.571, +0.435]
   * - 1
     - public-minus-parent
     - wp
     - -0.464 [-1.782, +0.854]
     - -0.310 [-1.016, +0.372]
   * - 1
     - public-minus-parent
     - point
     - +0.122 [-1.294, +1.514]
     - +0.180 [-0.567, +0.929]
   * - 1
     - masked-minus-parent
     - mixture
     - -0.598 [-1.538, +0.342]
     - -0.694 [-1.188, -0.206]
   * - 1
     - masked-minus-parent
     - wp
     - -0.220 [-1.538, +1.123]
     - -0.448 [-1.101, +0.221]
   * - 1
     - masked-minus-parent
     - point
     - -0.977 [-2.319, +0.367]
     - -0.939 [-1.649, -0.234]
   * - 2
     - public-minus-masked
     - mixture
     - +0.513 [-0.452, +1.477]
     - +0.253 [-0.262, +0.777]
   * - 2
     - public-minus-masked
     - wp
     - +0.415 [-0.928, +1.733]
     - -0.188 [-0.896, +0.507]
   * - 2
     - public-minus-masked
     - point
     - +0.610 [-0.781, +2.002]
     - +0.693 [-0.087, +1.491]
   * - 2
     - public-minus-parent
     - mixture
     - +0.720 [-0.183, +1.636]
     - +0.209 [-0.286, +0.710]
   * - 2
     - public-minus-parent
     - wp
     - +0.708 [-0.562, +1.953]
     - -0.119 [-0.769, +0.517]
   * - 2
     - public-minus-parent
     - point
     - +0.732 [-0.537, +2.026]
     - +0.537 [-0.223, +1.293]
   * - 2
     - masked-minus-parent
     - mixture
     - +0.208 [-0.732, +1.135]
     - -0.044 [-0.573, +0.475]
   * - 2
     - masked-minus-parent
     - wp
     - +0.293 [-1.025, +1.587]
     - +0.068 [-0.644, +0.767]
   * - 2
     - masked-minus-parent
     - point
     - +0.122 [-1.245, +1.489]
     - -0.156 [-0.938, +0.600]

Interpretation and next comparison
----------------------------------

Public-minus-masked mixture win changes are +0.427 and +0.513 percentage
points; public-minus-parent changes are -0.171 and +0.720. Every corresponding
interval crosses zero. The parent1 masked arm also loses expected points
relative to its parent in the mixture and Point strata. Those secondary point
results do not replace the failed win criterion. The intervals are per contrast;
no family-wide claim is assigned to the secondary cells.

The prespecified renewed-chance comparison now follows this completed study.
It retains public/masked controls and original parents while changing the
conditional deal at each training continuation. The motivation is finite
deal reuse; this result alone does not prove overfitting. The new trial was
designed before these outcomes and uses fresh 95M evaluation deals.

Validation and accounting
-------------------------

The independent audit reproduces all 18 win and point contrasts exactly,
replays every recorded full match, verifies matched initialization/warmup,
checks public-code provenance and 146,184 original historical inputs/masks.
It verifies 278 bound inputs and adds no neural inference or independent deal.

New training: 18,982,723 decisions (9,364,114 learner; 9,618,609 opponent),
266,240 conditional completions and 1,277,944 conditional round endings;
zero ordinary full training matches. Actual evaluation: 2,856,503 decisions,
191,108 round endings and 24,576 full matches. Public identification adds
201,568 strict predictions (100,192 WP; 101,376 Point). Independent evaluation
replay repeats those records and is separate work. Root reconstruction adds
212,212 recorded decisions, 14,101 round endings and 192 early match endings.

Evidence under runs/public_continuation_study:

* Protocol: 95ce0a01a76c790f5dbdeb0c78028199e61219053167fc7309543cfa4c4815a1.
* Independent audit: c1694086e2f91a7b13005324b409957cdd4aee4f84d09e2640e3e01a8601c322.
* Completed summary: 76f1352be16817c76e7c45fdeabf3cb97474581596809ded7cd50bc78a11eb29.

Producer46255 and independent queue96354 exited0. The summarizer ran once;
do not overwrite summary.md or summary_receipt.json or restart the finished
jobs. Initial dealer selection, the canonical game model and reserved101M
deals remain unchanged.
