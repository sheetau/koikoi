Public-memory final-round comparison: results
=============================================

Result
------

The predeclared balanced-mixture improvement criterion fails. Public memory
does not show a replicated match-win gain over either matched masked training
or the original parent. Every one of the18 reported win intervals and18 point
intervals includes zero. No model is promoted.

All four520-update training arms and24,576 strict full matches completed. The
two parents and three policies share2,048 fresh91M deal streams;opponent strata
contain1,024 paired deals each. The primary criterion requires positive lower
95% bounds for public-minus-masked and public-minus-parent mixture wins for both
parents. Opponent-specific and masked-minus-parent contrasts are secondary.

Full contrasts
--------------

.. list-table:: Paired changes with95% confidence intervals
   :header-rows: 1

   * - Parent
     - Contrast
     - Scope
     - Win change (pp)
     - Point change
   * - 1
     - public-minus-masked
     - mixture
     - -0.256 [-0.537, +0.024]
     - -0.037 [-0.141, +0.067]
   * - 1
     - public-minus-masked
     - wp
     - -0.293 [-0.684, +0.098]
     - +0.010 [-0.136, +0.157]
   * - 1
     - public-minus-masked
     - point
     - -0.220 [-0.610, +0.171]
     - -0.083 [-0.231, +0.064]
   * - 1
     - public-minus-parent
     - mixture
     - -0.122 [-0.439, +0.183]
     - -0.079 [-0.188, +0.026]
   * - 1
     - public-minus-parent
     - wp
     - -0.024 [-0.488, +0.439]
     - -0.039 [-0.196, +0.119]
   * - 1
     - public-minus-parent
     - point
     - -0.220 [-0.659, +0.195]
     - -0.118 [-0.270, +0.026]
   * - 1
     - masked-minus-parent
     - mixture
     - +0.134 [-0.195, +0.464]
     - -0.042 [-0.162, +0.076]
   * - 1
     - masked-minus-parent
     - wp
     - +0.269 [-0.195, +0.732]
     - -0.049 [-0.207, +0.110]
   * - 1
     - masked-minus-parent
     - point
     - +0.000 [-0.488, +0.464]
     - -0.035 [-0.218, +0.139]
   * - 2
     - public-minus-masked
     - mixture
     - +0.037 [-0.244, +0.317]
     - +0.047 [-0.062, +0.158]
   * - 2
     - public-minus-masked
     - wp
     - -0.220 [-0.659, +0.220]
     - -0.027 [-0.186, +0.140]
   * - 2
     - public-minus-masked
     - point
     - +0.293 [-0.049, +0.659]
     - +0.122 [-0.029, +0.272]
   * - 2
     - public-minus-parent
     - mixture
     - +0.073 [-0.232, +0.378]
     - +0.033 [-0.075, +0.142]
   * - 2
     - public-minus-parent
     - wp
     - -0.122 [-0.586, +0.342]
     - -0.028 [-0.191, +0.134]
   * - 2
     - public-minus-parent
     - point
     - +0.269 [-0.122, +0.684]
     - +0.095 [-0.049, +0.238]
   * - 2
     - masked-minus-parent
     - mixture
     - +0.037 [-0.305, +0.378]
     - -0.014 [-0.139, +0.111]
   * - 2
     - masked-minus-parent
     - wp
     - +0.098 [-0.415, +0.586]
     - -0.001 [-0.185, +0.178]
   * - 2
     - masked-minus-parent
     - point
     - -0.024 [-0.488, +0.439]
     - -0.027 [-0.195, +0.139]

Work and independent verification
---------------------------------

Training totals3,725,648 decisions:1,837,832 learner and1,887,816 opponent
decisions,266,240 conditional final-round completions,and zero full training
matches. Source generation and public identification reuse the preceding
training-history artifacts;they add no new source deals here.

Evaluation executes2,857,337 decisions,191,262 round endings and24,576 complete
matches. Online public-model elimination makes355,486 strict predictions
(167,506 WP and187,980 Point). Independent replay repeats the recorded matches
and verifies300,092 historical feature/mask reconstructions against the original
builder. It performs zero new neural predictions or independent matches.

The independent audit also verifies all initial/warmup states,public-code joins,
frozen parent/opponents,Adam states and full paired/bootstrap statistics. All181
audit input hashes reverify. Producer67506 and audit queue35738 terminate exit0;
the summarizer then executes once successfully. Do not restart completed runs.
Short CPU diagnostics/tests overlapped evaluation;timings are not isolated
latency benchmarks.

Evidence
--------

Protocol:3e1282183196b0f76bfbaa383160304f34377c12d482578bf4cec112a627c7c1.
Producer analysis:10c018a55a2fd006969aea819e207a59dd5c0f6a4b54241266727c01f655a8f9.
Independent audit:a5a9f350f2388294cdd106758d60eb453a2f03a24ff41f192436a04d4925acd7.
Summary:ca6182fb9e687e4c0a23237e161a4bc27de518ffb40e12b32c9b650e583fc985.
Summary receipt:a566ffd3a66a755d6ff260750a67f909a7cab426d1ad1550bb86f9db8e26c9a9.

Artifacts are in``runs/public_memory_study``;the frozen design is in
PUBLIC_MEMORY_STUDY.rst. The canonical playable model,dealer and reserved101M
confirmation deals remain unchanged.

Next direction
--------------

The separate training-history diagnostic found1,731 usable fresh-round roots
after public identification,including1,642 before the final round. A continuation
learner and online adapter now cover all remaining rounds after that boundary.
This provides an earlier adaptation opportunity;its strength remains untested.
Its subsequent comparison must use a fresh fixed protocol and deals,retaining
the masked control and all unsuccessful identification cases.
