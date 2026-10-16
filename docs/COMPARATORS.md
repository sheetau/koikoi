# Public comparator review

Reviewed 2026-10-17. This is a bounded review of relevant public implementations, not an exhaustive ranking. No additional opponent was evaluated in this review. Repository snapshots are in ignored `external/`; no upstream source or artwork was incorporated into our package.

## Primary strength gate

Keep the pinned [Guan et al. KoiKoi-AI](https://github.com/guansanghai/KoiKoi-AI/tree/89d789565c7c36c84b74662e43001d960bf80901) assisted RL-WP opponent as the primary gate. Our `paper8` rules reproduce that implementation; `research12` changes match length and bankruptcy and is a separate experiment. The associated paper is [Learning to Play Koi-Koi Hanafuda Card Games With Transformers](https://ieeexplore.ieee.org/document/10032777/), DOI 10.1109/TAI.2023.3240674. See RULES.md for the actual scoring contract and EXPERIMENTS.md for measured comparisons.

Beating this opponent would satisfy the named minimum baseline, not establish universal superiority over every Koi-Koi implementation. Preserve independent-model validation, untouched final deals and disclosure of development selection.

## Open Hanafuda AI

Pinned snapshot: `suzukiplan/open-hanafuda-ai` at `01b4e484ac3d223f4ae8198d2a399f69078174d2`; remote HEAD and local checkout agree. Its [README](https://github.com/suzukiplan/open-hanafuda-ai/blob/01b4e484ac3d223f4ae8198d2a399f69078174d2/README.md) describes simulation, observable-information inference and heuristics, with MIT-licensed code.

The documented rules differ materially: one koi declaration per round, seven-point and opponent-koi doubling, initial-hand automatic outcomes, and a scoreless exhausted-hand result with dealer switching. Our paper-derived profiles have different declaration scoring and exhausted-hand outcomes. The README's claim of Nintendo conformity has not been independently audited here. Its example Hard-versus-Normal result is an internal comparison, not a comparison with assisted RL-WP.

Disposition: useful algorithmic reference and potential additional comparator, but not a drop-in opponent. A faithful port must explicitly reconcile scoring, legal actions, initial deals, card IDs and hidden-information boundaries. Changing its rules creates an adapted opponent and must be labelled accordingly. Do not infer its strength against our agents from its internal benchmark.

## Koikoi Swift

Pinned snapshot: `ngs/koikoi-swift` at `c8de6e1207ca4cdaf29d52cdd631fc26d4583538`; remote HEAD and local checkout agree. Inspected the actual [ISMCTS source](https://github.com/ngs/koikoi-swift/blob/c8de6e1207ca4cdaf29d52cdd631fc26d4583538/Sources/AI/ISMCTS.swift), rather than relying on the application description.

It implements single-observer determinization, availability-based UCB selection, heuristic rollouts and selection by root visit count. Search terminates at a round outcome; reward scales and clips round points. The configuration defaults to 800 iterations; the UI instantiates 400. These are different from our complete-match rollout objective and particle budget.

[Game.calcFinalScore](https://github.com/ngs/koikoi-swift/blob/c8de6e1207ca4cdaf29d52cdd631fc26d4583538/Sources/Core/Game.swift) applies seven-point and opponent-koi doubling, already sufficient to establish incompatibility with paper8. This was source inspection, not a build, exhaustive rules audit or strength evaluation. Source code is MIT-licensed; artwork is excluded under its [license](https://github.com/ngs/koikoi-swift/blob/c8de6e1207ca4cdaf29d52cdd631fc26d4583538/LICENSE).

Disposition: a reference for a separately specified tree-search experiment. Do not replace the current fixed full-action endpoint or call an adapted round-utility search a reproduction of its original playing strength.

## Experimental consequence

Finish the already committed 1,000-pair full-action development endpoint unchanged. If the effect persists, validate the same method on the independent existing model before final testing. Any additional comparator needs a pinned implementation, explicit rule compatibility, observation-only adapter checks and paired deals. Cross-variant results must remain separate; compute scale alone is not evidence of superiority.
