Public revelation and opponent adaptation: closest prior work
============================================================

Primary sources checked on 2026-10-19
------------------------------------

Southey et al., Bayes' Bluff (UAI 2005), Section3.3, gives separate
observation likelihoods for showdown and folded hands. Showdowns permit
conditioning on revealed private cards; folds require marginalizing the
unobserved cards. The resulting strategy posterior supports a response.
Therefore retrospectively using revealed cards in opponent modeling is
established prior work, not a new general inference principle here.

Author-hosted paper:
https://webdocs.cs.ualberta.ca/~mbowling/papers/05uai.pdf

Ganzfried, Wang and Chiswick, Opponent Modeling in Multiplayer
Imperfect-Information Games (DAI 2024), Sections1--3 and Algorithm2,
explicitly consider partial revelation between repeated stage games.
They collect observations under a default policy, then use responses to
updated opponent models. This is a close precedent for collecting
information before later adaptation. Their sampled Bayesian strategy
models and switching schedule differ from our finite deterministic
candidate elimination and public identification boundary. The conference
PDF contains placeholder DOI/ISBN fields; the bibliography deliberately
does not copy those placeholders. Its author list, title, venue and year
are present on the paper. A 2022 arXiv version also exists.

Conference-hosted paper:
https://www.adai.ai/dai/2024/dai_papers/DAI2024_paper_10.pdf

Rebstock et al., Policy Based Inference in Trick-Taking Card Games
(IEEE CoG 2019), SectionIII and Algorithm1, use action probabilities from
a player model to weight possible hidden deals in Skat. The inference
target is the hidden state under a given model. Our completed revelation
diagnostic instead reconstructs historical network inputs after all eight
hand discards and tests which fixed opponent policies match past actions.
The distinction concerns the inference target and information available,
not a claim to have invented action-likelihood inference.

Author-submitted full text:
https://arxiv.org/html/1905.10911v1

Implications for the current contribution
---------------------------------------

The paper now describes an exact Koi-Koi input-reconstruction implementation,
its measured identification coverage, and a controlled test of downstream
benefit. These are supported by the recorded input/mask equivalence checks,
public-availability timestamps and matched public/masked comparisons.
General opponent modeling, delayed revelation and a default-policy prefix
are not presented as new ideas. Whether earlier use improves full-match
performance remains the question of the running prospective study; neither
identification accuracy nor prior-art differentiation establishes that gain.

This literature clarification changes no live-study protocol, source,
training budget, evaluation seed, selection criterion or deployment model.
