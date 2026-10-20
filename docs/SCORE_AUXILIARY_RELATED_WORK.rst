Auxiliary learning: prior work and manuscript placement
======================================================

Primary sources inspected on 20 October 2026. The main six-condition strength
comparison is running under its frozen protocol. This literature review changes
neither that protocol nor its criteria. The verified manuscript PDF remains the
previous version; the two new manuscript fragments are integration drafts.

The question to carry into the paper is whether supervision that distinguishes
margins within the same match outcome improves a policy optimized for match wins.
The matched 65-category head keeps parameterization and initial outcome masses
fixed. Adding the within-outcome likelihood changes both target information and
optimization. This comparison can establish the effect of that intervention;
it cannot separate those two explanations by itself.

Direct foundations
------------------

* UNREAL, Jaderberg et al.: shared representations trained by auxiliary control
  and prediction tasks. This establishes the broader auxiliary-learning idea.
  Source: https://arxiv.org/abs/1611.05397v1
* Phasic Policy Gradient, Cobbe et al., ICML 2021, Section 2: alternates policy
  and auxiliary optimization with policy cloning during the auxiliary phase.
  Section 3.4 also studies the effect of phase frequency. Our phase separation
  and KL penalty follow this framework; terminal-margin supervision is the
  specific target tested here.
  Source: https://proceedings.mlr.press/v139/cobbe21a/cobbe21a.pdf
* Distributional RL, Bellemare et al., ICML 2017: predicts distributions with
  Bellman updates. Our 65-bin prediction uses completed Monte Carlo margins,
  so it should be described as categorical auxiliary supervision, not C51.
  Source: https://proceedings.mlr.press/v70/bellemare17a.html

Alternative explanations to preserve
------------------------------------

* Imani and White, ICML 2018, analyze distributional regression losses and
  support an optimization explanation through gradient behavior. This motivates
  separating a claim about a successful loss from a claim about better
  representations. Source: https://proceedings.mlr.press/v80/imani18a.html
* Lyle, Bellemare and Castro, AAAI 2019, Sections 5-6, study nonlinear
  approximation, early-layer learning and categorical outputs. This is direct
  precedent for studying why distributional targets change learning. The head
  comparison here controls output size but not all gradient effects.
  Source: https://arxiv.org/html/1901.11084v2
* Garcin et al., ICLR 2025, Sections 4-6, investigate actor/critic specialization,
  auxiliary objectives and training-data diversity. Their experiments show why
  value-related information is not automatically useful to an actor. Section 5
  links benefits from value distillation to large, diverse training batches.
  This motivates keeping target choice, data diversity and action relevance as
  distinct explanations when interpreting our results. It does not diagnose a
  cause in Koi-Koi without further evidence.
  Source: https://arxiv.org/html/2503.06343v2

Integration materials
---------------------

* docs/paper/auxiliary_related_work.tex: concise three-paragraph subsection.
* docs/paper/auxiliary_references.bib: six primary-source references, with
  bibliographic metadata checked against the publisher or authors' record.
* docs/SCORE_AUXILIARY_MECHANISM_RESULTS.rst: completed fixed-state evidence.

Merge the subsection and references when adding the completed strength study
and its methods to the manuscript. The current draft makes no strength claim
and introduces no placeholder result. After integration, rebuild and inspect
the PDF, including citations and pagination. Current manuscript.tex,
references.bib and the verified 35-page PDF are unchanged.

The stronger-play objective remains open. Auxiliary phases, categorical
prediction and the likelihood chain rule are established tools. A contribution
from this study requires the actual controlled playing result and a clear
account of what the target-resolution comparison establishes. The source review
adds relevant precedents; it is not a proof of worldwide priority.
