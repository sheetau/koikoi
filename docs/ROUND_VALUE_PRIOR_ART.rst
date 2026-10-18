Koi-Koi continuation-table prior art and claim boundary
======================================================

Direct source inspection
------------------------

The pinned Guan et al. implementation already uses a public continuation-win
table. This is not only a generic connection to conditional Monte Carlo.
The manuscript's Related Work already acknowledges the table; the new round
search documents must be read with the same attribution.

Pinned upstream revision:89d789565c7c36c84b74662e43001d960bf80901.

* rl_train.py loads win_prob_mat.pkl. TraceSimulator.__reward_wp indexes it by
  next round,player's updated points,and whether the round winner is that player.
  Terminal draws/wins use0.5/1.0 and rewards are multiplied by10.
  https://github.com/guansanghai/KoiKoi-AI/blob/89d789565c7c36c84b74662e43001d960bf80901/rl_train.py
* rl_wp_reward_simulate.py describes self-play continuation matches over rounds
  1..8,points1..59,with dealer1 and200 tests per context in its main loop.
  https://github.com/guansanghai/KoiKoi-AI/blob/89d789565c7c36c84b74662e43001d960bf80901/rl_wp_reward_simulate.py
* rl_wp_reward_regress.py fits a separate logistic regression of win outcome on
  points for each round,and obtains the other dealer slice by complementary
  point reversal. It writes win_prob_mat_new.pkl.
  https://github.com/guansanghai/KoiKoi-AI/blob/89d789565c7c36c84b74662e43001d960bf80901/rl_wp_reward_regress.py

This inspection does not reproduce the historical table-generation run. In
particular,the released simulation call's argument tuple differs from the
function signature,and the regression output filename differs from the trainer's
input. Do not claim that this inspected pipeline exactly regenerated the shipped
win_prob_mat.pkl or infer a defect in its historical experimental outcomes.

SHA256 identities:rl_train.py
4b1ecbcf71bdf779c010e861e5758004db74f1939e85a64ed41dc3b9dad01046;
rl_wp_reward_simulate.py
8b08bba55d7791cf4ebd4848d44dc720b50ca22d78e88bf6d054995f249e0b51;
rl_wp_reward_regress.py
6e34eb825c10f963c73fb2e9e981e431976ab38f40282da3492332def80c0004;
win_prob_mat.pkl
e39be1a75a83813c96835d13147fdaa5c4ae31fdcee1ca177d046fda05b6a452.

What the new implementation changes
-----------------------------------

The new artifact evaluates single-round transition kernels backward under the
retained parent and each fixed WP/Point continuation. It keeps absolute parent
seat and dealer axes,uses every live score without logistic interpolation,and
preserves shared-deal samples for uncertainty analysis. Independent A/B tables
separate candidate proposal from confirmation. History-conditioned particles
retain the joint hand/model posterior rather than supplying actual opponent
identity. These ingredients are used for inference-time paired action comparison,
with table error propagated jointly across particles.

Boundary tables,Bayesian conditioning,truncated rollout,backward policy
evaluation and bootstrap resampling are established ideas. The above differences
define an intervention and ablations;they are not by themselves a new general
algorithm or an established advantage over the original work. A publishable
contribution must identify a concrete information/estimation problem and show
an independently measured accuracy,cost or policy benefit. Match-strength
claims still require complete matches and parent replication.

Do not describe the new work as introducing continuation tables to Koi-Koi.
Retain the original architecture/adaptation and critic-information results
under their actual measured scope while the new estimator/search studies run.
