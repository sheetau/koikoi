Point initialization calibration: completed result
=================================================

The four fixed Point sampling divisors are:

* Discard:0.16808073347585306
* Hand capture:0.19483521603032863
* Draw capture:0.21949649006919622
* Stop/continue:0.23576122751596976

Each matches WP's phase-average normalized entropy at0.05 on75old calibration
source groups. The separate25old diagnostic groups are descriptive validation,
not fresh held-out games. Their mean normalized entropies are:

===============  ========  ===========  =============
Phase            WP0.05    Point0.05    Point matched
===============  ========  ===========  =============
Discard          0.452271  0.153910     0.379238
Hand capture     0.710939  0.349895     0.640632
Draw capture     0.611028  0.303543     0.579449
Stop/continue    0.199169  0.028503     0.130684
===============  ========  ===========  =============

Differences remain on validation. The fitted values are retained unchanged;
matching calibration means does not make the policies equally exploratory on
all states, in training, or after updates. All original calibrated targets and
row counts are retained in the report. Calibrated rows per phase:7641/357/344/638;
validation:2578/122/130/215. Guards remove forced endgame choices. Negative
source IDs are excluded and no source group belongs to both partitions.

Independent scalar softmax/entropy reproduces all24reported means within1e-12.
Three tests pass0skips:known-scale recovery/group filtering,validation and
ineligible-score independence,andunattainable-target refusal. Ruff passes.
Test XML runs/point_initialization_calibration_tests.xml:
dc7e7bdf87749dc828cec920f4a5010e810edae55b01b3271e3d3de791b0a32e.

Report runs/point_initialization_calibration.json:
869ab1e45e7e8a5569ff20edaea55fdc5e2a2464d6d6a32a9c50df188bd701f3,
binding9inputs. No outcomes are read;new neural calls,simulator decisions,
training updates and matches are allzero. Calibration is not a strength result.

Next implement and capability-test the isolated four-phase temperature learner.
The main seeds,schedule and decision rules remain to be fixed before training.
The prior Point initialization diagnostic remains unchanged and the completed
duration study is closed. See POINT_INITIALIZATION_STUDY.rst for the fixed
calibration procedure and intended three-arm comparison.
