Prior-work notes for public score-bound planning
===============================================

Primary sources checked on 22 October 2026. These notes support a later
manuscript revision after the fixed development screen closes; no pending
experiment is described as a measured strength improvement.

Chance-node bounds
------------------

Ballard, B. W. (1983), The star-minimax search procedure for trees containing
chance nodes (title uses the asterisk symbol), Artificial Intelligence 21(3), 327--350.
DOI: https://doi.org/10.1016/S0004-3702(83)80015-0.
Primary paper: https://www.cs.uleth.ca/~benkoczi/3750/data/ballard83-star_alpha_beta.pdf.

Sections 1--4 explicitly treat perfect-information games with chance nodes.
Explored successors and valid leaf-value limits bound the weighted chance-node
value; those limits permit cutoffs and tighter windows below chance nodes.
This is direct precedent for using unresolved probability mass to bound an
unfinished comparison. Neither chance-node bounds nor preserving a completed
incumbent should be presented as a new general search principle here.

Our contribution must be stated at the implemented domain boundary: public
captured-card exclusions and remaining capture opportunities yield Koi-Koi
score limits; score limits imply policy-independent match outcomes; the saved
program remains executable through a checked parent default. Decisions still
share an information-set action across compatible hidden deals. That last
constraint is absent from the perfect-information formulation in Ballard.

Statistical selective pruning
-----------------------------

Buro, M. (1995), *ProbCut: An Effective Selective Extension of the Alpha-Beta
Algorithm*, ICCA Journal 18(2), 71--76.
Primary publisher abstract and metadata:
https://journals.sagepub.com/doi/10.3233/ICG-1995-18202.

The publisher abstract describes using the correlation between shallow and
deep evaluations to predict a search-window crossing at a prescribed
likelihood. It reports an Othello application. Only the abstract/metadata were
available here; do not attribute uninspected implementation details.

Our outcome shortcut is instead derived from legal scoring and public state.
Its correctness has no fitted confidence parameter. Population uncertainty
about an average gain remains a separate evaluation question. This difference
explains the guarantee; it does not by itself establish practical superiority
over ProbCut or another selective-search implementation.

Suggested integration
---------------------

Add a short chance-node pruning paragraph to the existing search subsection
and cite these precedents where bounded continuation and the new public score
bound are introduced. State the actual Koi-Koi construction first, followed
by the required information-set and executable-policy conditions. Keep timing,
completed references, adopted policies and match-win changes as separate
outcomes. Do not expand the Abstract with every implementation variant.
