Point initialization with an explicit exploration control
=========================================================

Motivation and boundary
-----------------------

The completed duration comparison improves mixture point margins but fails its
replicated additional-win criterion. Preserve it without extra training or deals.
The archived initialization comparison finds much lower initial Point entropy
than WP at the same score divisor. Existing recorded pretrained-PPO configurations
all start from WP. Testing Point is therefore an untested initialization choice,
not a new algorithm. The retained player and dealer remain unchanged.

Fixed calibration specification
-------------------------------

Before any new learning, use the existing archived CPU WP/Point scores in the
pinned koi_temperature_mechanism corpus. Only nonnegative source IDs and decisions
with at least two legal actions after public endgame assistance are eligible.
Source ID modulo4 !=3 supplies calibration, ==3 supplies descriptive validation.
These groups have been used in earlier diagnostics; they are not fresh holdout.
No outcomes or hidden-card inputs enter this calculation.

For each of the four phases separately, fix the target to WP's mean entropy
divided by log(legal-action count), at divisor0.05 on calibration rows. Solve one
positive Point divisor to match that target using the existing70-step log-space
bisection on[1e-5,100]. Reject unattainable targets or missing phase groups.
Freeze all four results without choosing among validation outcomes. Report
WP-fixed,Point-fixed andPoint-matched on both partitions and independently check
the entropy by scalar arithmetic. This calibration adds no network inference,
game transitions or training. It does not match complete sampling distributions,
gradient magnitudes, future entropy or training visitation.

Planned controlled learning
---------------------------

Use three arms at two fresh seeds:WP with0.05,Point with0.05,andPoint with the
four calibrated divisors. Each uses the unchanged architecture,match reward,
public critic,20critic-only warmup and500policy updates. Preserve the same
optimizer settings,production128x128rollouts,512minibatches,threeepochs and
float32CUDA. The calibrated divisors are constant throughout training; greedy
deployment remains the published raw-score action ranking.

Point-fixed versusWP-fixed measures the literal initialization replacement.
Point-matched versusPoint-fixed measures the additional temperature intervention.
Point-matched versusWP-fixed compares initializations with matched phase-average
calibration entropy. This last contrast does not isolate all differences in
exploration or effective optimization. A later mechanism assessment must check
on-policy entropy and actual policy changes rather than assume matching persists.

Before main training, implement an isolated phase-temperature wrapper and verify
that the all0.05control exactly reproduces PublishedTrainer,that all arms use
their declared original weights,andthat save/restore preserves the next update.
Check critic-only actor stationarity,actor movement after warmup,public-only
features and deterministic greedy inference. Use separate capability seeds and
count repeated updates separately. Keep earlier frozen sources unchanged.

Main seed ranges,evaluation schedule,multiplicity family and qualification rule
must be frozen after capability and seed-inventory checks but before any main
training/evaluation. No main study is launched by this calibration specification.
The development criterion must test improvement at both training seeds and
against both the matched WP arm and retained players. Report all arms and
opponent-specific wins/points; do not promote by a favorable single seed.
Reserved101Mconfirmation remains unused and separate. Stronger play and method
novelty both remain open; no claim follows merely from changing initialization.
