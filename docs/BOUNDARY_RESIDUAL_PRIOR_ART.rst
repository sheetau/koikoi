Boundary-residual estimator: related work and contribution boundary
==================================================================

The implementation in boundary_residual_values.py uses established estimation
ideas. Two directly relevant primary sources were checked on 2026-10-19:

* Peherstorfer, Willcox and Gunzburger (2016), Optimal Model Management for
  Multifidelity Monte Carlo Estimation, SIAM Journal on Scientific Computing
  38(5), A3163--A3194, DOI 10.1137/15M1046472.
  https://dspace.mit.edu/entities/publication/f104683d-44c4-40ec-bd06-47502c5c0bfd
  The paper combines surrogate and high-fidelity evaluations for unbiased
  estimation and optimizes their allocation. Our fixed-coefficient two-level
  identity is not a new general estimation result.

* Veness, Lanctot and Bowling (2011), Variance Reduction in Monte-Carlo Tree
  Search, Advances in Neural Information Processing Systems 24.
  https://papers.neurips.cc/paper_files/paper/2011/hash/d736bb10d83a904aefc1d6ce93dc54b8-Abstract.html
  This paper studies common random numbers, antithetic variates and control
  variates in stochastic game planning. Consequently, applying variance
  reduction to game simulation is not itself a novelty claim either.

The Koi-Koi implementation has a narrower purpose. It combines the checked
public-history hand/model distribution with an existing public round-boundary
proxy and a terminal residual under an explicit numerical continuation. It
fixes the first-round execution in both fidelities and corrects the proxy's
approximation error relative to that target. The diagnostic retains uniform
and publicly identified/history-conditioned distributions, independent reference
samples and measured work. These define testable differences from the earlier
Koi-Koi prototypes, not an established advantage.

An appropriate research contribution would require a measured estimation-cost
benefit, useful action targets or stronger complete-match play. The completed
50-cell development diagnostic found no variance-cost benefit at its fixed
allocation; see BOUNDARY_RESIDUAL_RESULTS.rst. Adding
these sources to a manuscript is appropriate if this method becomes part of
the paper; a standalone prototype should not inflate the main Related Work or
be presented as a new algorithm solely through terminology.
